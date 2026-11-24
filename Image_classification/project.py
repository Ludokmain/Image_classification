import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
 
import torch
import torchvision
import sklearn
import platform
import json
import joblib
import matplotlib.pyplot as plt
import copy
import time
import torch.nn as nn
import random, numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
 
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
 
SEED = 42
NUM_CLASSES = 37
VAL_FRACTION = 0.2
NUM_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 0.01  # valore di default di AdamW, scritto esplicitamente per documentarlo
LOGREG_C_GRID = [0.001, 0.01, 0.1, 1]
SVM_C_GRID = [0.001, 0.01, 0.1, 1]
LOGREG_MAX_ITER = 5000
SVM_MAX_ITER = 10000
N_AUG_COPIES = NUM_EPOCHS  # copie aumentate del training set per i classificatori sklearn: tante quante le epoche delle reti
N_BOOTSTRAP = 1000
best_val_acc = 0
best_weights = None
 
 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# warn_only=True: se un'operazione non ha una versione deterministica, PyTorch avvisa invece di fermarsi
torch.use_deterministic_algorithms(True, warn_only=True)
 
 
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
 
 
print(f"Using device: {device}\n")
 
 
def sync():
    # le GPU lavorano in modo asincrono: senza sincronizzare, time.time() misurerebbe
    # solo il tempo per "mettere in coda" le operazioni, non quello per eseguirle
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
 
 
weights = models.MobileNet_V3_Small_Weights.DEFAULT
 
# preprocessing ufficiale dei pesi: Resize(256) -> CenterCrop(224) -> Normalize
eval_transform = weights.transforms()
 
 
# augmentation solo per il training, con la stessa normalizzazione dei pesi
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(eval_transform.crop_size[0]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=eval_transform.mean, std=eval_transform.std),
])
 
# download del dataset, due copie di trainval: una per il training con augmentation, una per la valutazione senza augmentation
trainval_dataset_aug = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=train_transform)
trainval_dataset_eval = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=eval_transform)
test_dataset = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=eval_transform)
 
 
# split stratificato per razza, ogni razza divisa tra train e validation mantenendo le proporzioni
labels = np.array(trainval_dataset_eval._labels)
indices = np.arange(len(labels))
train_idx, val_idx = train_test_split(
    indices,
    test_size=VAL_FRACTION,
    stratify=labels,
    random_state=SEED,
)
 
np.savez("splits.npz", train_idx=train_idx, val_idx=val_idx)  # salvo gli indici di train e validation
 
 
train_set = Subset(trainval_dataset_aug, train_idx)
val_set = Subset(trainval_dataset_eval, val_idx)
train_set_eval = Subset(trainval_dataset_eval, train_idx)  # stesse immagini del train_set ma senza augmentation
 
# training set: per allenare il modello con data augmentation
# validation set: per valutare il modello durante il training e scegliere checkpoint/iperparametri
 
 
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
train_eval_loader = DataLoader(train_set_eval, batch_size=BATCH_SIZE, shuffle=False)
 
# conteggio immagini per razza in ogni split
class_names = trainval_dataset_eval.classes
train_counts = Counter(labels[i] for i in train_idx)
val_counts = Counter(labels[i] for i in val_idx)
test_counts = Counter(test_dataset._labels)
 
 
print(f"{'Breed':<30}{'Train':>7}{'Val':>7}{'Test':>7}")
for c, name in enumerate(class_names):
    print(f"{name:<30}{train_counts[c]:>7}{val_counts[c]:>7}{test_counts[c]:>7}")
print(f"{'TOTAL':<30}{len(train_idx):>7}{len(val_idx):>7}{len(test_dataset):>7}")
 
# salvo la tabella dei conteggi per il report e il repository
with open("class_counts.csv", "w") as f:
    f.write("breed,train,val,test\n")
    for c, name in enumerate(class_names):
        f.write(f"{name},{train_counts[c]},{val_counts[c]},{test_counts[c]}\n")
 
# sbilanciamento: rapporto tra la classe più numerosa e la meno numerosa (1.0 = perfettamente bilanciato)
imbalance = {
    "train": max(train_counts.values()) / min(train_counts.values()),
    "val": max(val_counts.values()) / min(val_counts.values()),
    "test": max(test_counts.values()) / min(test_counts.values()),
}
print("\nClass imbalance ratio (max/min): " + ", ".join(f"{k} {v:.2f}" for k, v in imbalance.items()))
print("Nessun class-balancing applicato (niente pesi nella loss, niente oversampling)\n")
 
 
################################ funzioni comuni di valutazione ################################
 
results = []  # una riga per metodo, diventa la tabella dei risultati
histories = {}  # curve di training per epoca delle due reti
 
 
def count_trainable(model):
    # numero di parametri con requires_grad=True, cioè quelli aggiornati dall'optimizer
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
 
 
def predict_nn(model, loader):
    # restituisce predizioni ed etichette vere per tutto il loader
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)
 
 
def bootstrap_ci(y_true, y_pred, n=N_BOOTSTRAP):
    # intervallo di confidenza al 95% della test accuracy:
    # ricampiono il test set con reinserimento n volte, calcolo l'accuracy di ogni campione
    # e prendo i percentili 2.5 e 97.5 della distribuzione ottenuta
    rng = np.random.default_rng(SEED)
    accs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    low, high = np.percentile(accs, [2.5, 97.5])
    return round(100 * low, 2), round(100 * high, 2)
 
 
def save_confusion_matrix(y_true, y_pred, title, filename):
    # labels=range(NUM_CLASSES) garantisce una matrice 37x37 anche se una classe non viene mai predetta
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(16, 16))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    return cm
 
 
def top_confusions(cm, k=5):
    # coppie (vera, predetta) più frequenti fuori dalla diagonale
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    pairs = []
    for f in flat:
        i, j = np.unravel_index(f, off.shape)
        pairs.append((class_names[i], class_names[j], int(off[i, j])))
    return pairs
 
 
def add_result(method, y_train, train_pred, y_test, test_pred, best_val, n_params, train_time, selected, cm_file):
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average="macro")
    ci_low, ci_high = bootstrap_ci(y_test, test_pred)
    cm = save_confusion_matrix(y_test, test_pred, f"{method} - test", cm_file)
    confusions = top_confusions(cm)
    r = {
        "method": method,
        "selected": selected,
        "best_val": best_val,
        "train_acc": round(100 * train_acc, 2),
        "test_acc": round(100 * test_acc, 2),
        "test_acc_ci95": [ci_low, ci_high],
        "test_macro_f1": round(test_f1, 4),
        "train_test_gap": round(100 * (train_acc - test_acc), 2),
        "trainable_params": int(n_params),
        "train_time_s": round(train_time, 1),
        "top_confusions": confusions,
    }
    results.append(r)
    print(f"[TEST] {method}: acc {r['test_acc']:.2f}% (95% CI {ci_low:.2f}-{ci_high:.2f}), "
          f"macro-F1 {r['test_macro_f1']:.4f}, train acc {r['train_acc']:.2f}%, gap {r['train_test_gap']:.2f} pp")
    print("Most confused pairs (true -> predicted, count):")
    for t, p, n in confusions:
        print(f"   {t} -> {p}: {n}")
    print()
 
 
def show_examples(dataset, y_true, y_pred, title, filename, n=8):
    # griglia: prima riga predizioni corrette (verde), seconda riga errori (rosso)
    rng = np.random.default_rng(SEED)
    correct_idx = np.where(y_pred == y_true)[0]
    wrong_idx = np.where(y_pred != y_true)[0]
    sel_c = rng.choice(correct_idx, min(n, len(correct_idx)), replace=False)
    sel_w = rng.choice(wrong_idx, min(n, len(wrong_idx)), replace=False)
    # per visualizzare l'immagine annullo la normalizzazione: x_originale = x_norm * std + mean
    mean = torch.tensor(eval_transform.mean).view(3, 1, 1)
    std = torch.tensor(eval_transform.std).view(3, 1, 1)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6))
    for row, (sel, color) in enumerate([(sel_c, "green"), (sel_w, "red")]):
        for ax in axes[row]:
            ax.axis("off")
        for ax, i in zip(axes[row], sel):
            img, _ = dataset[i]  # test_loader ha shuffle=False, quindi l'indice i corrisponde all'immagine i del dataset
            img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()  # da (C,H,W) a (H,W,C) per imshow
            ax.imshow(img)
            ax.set_title(f"T: {class_names[y_true[i]]}\nP: {class_names[y_pred[i]]}", fontsize=7, color=color)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
 
 
def plot_history(history, title, filename):
    # due grafici affiancati: loss e accuracy per epoca, train vs validation
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], marker="o", label="train")
    ax1.plot(epochs, history["val_loss"], marker="o", label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], marker="o", label="train (with augmentation)")
    ax2.plot(epochs, history["val_acc"], marker="o", label="validation")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")
    ax2.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)
 
 
# definizione della funzione per costruire il modello
def build_model():
    torch.manual_seed(SEED)  # seed per la riproducibilità
    m = models.mobilenet_v3_small(weights=weights)
    # sostituisco l'ultimo layer con un nuovo layer lineare a 37 uscite
    num_features = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(num_features, NUM_CLASSES)
    return m.to(device)
 
###################################### full fine-tuning ############################################
 
model1 = build_model()
optimizer = torch.optim.AdamW(model1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()
best_epoch = None
history1 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
 
 
print("\nFULL FINE-TUNING\n")
 
sync()
t0 = time.time()
 
for epoch in range(NUM_EPOCHS):
    model1.train()
    running_loss = 0.0  # somma della loss di tutte le immagini viste finora
    train_correct = 0  # numero di immagini predette correttamente finora
    train_total = 0  # numero totale di immagini viste finora
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # azzera i gradienti (accumulati)
        # forward pass
        outputs = model1(images)
        loss = criterion(outputs, labels)  # forward pass + loss
        loss.backward()  # backpropagation
        optimizer.step()  # update dei pesi
        running_loss += loss.item() * images.size(0)  # loss è un tensore con un solo elemento, la media della loss del batch, .item() estrae il valore scalare, images.size(0) è il numero di immagini nel batch
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()  # argmax restituisce la posizione del valore maggiore,
        # dim=1 indica la dimensione lungo cui cercare: la dimensione 1 sono le colonne, cioè le classi.
        # Quindi cerca il massimo in ogni riga, cioè per ogni immagine, si ottiene una sequenza di indici, che sono le classi predette.
        # ==labels confronta con le etichette vere, restituendo True se uguali, False altrimenti
        train_total += labels.size(0)  # quante immagini abbiamo visto finora
 
    train_loss = running_loss / train_total  # loss media per immagine per epoca
    train_acc = 100.0 * train_correct / train_total  # percentuale di immagini predette correttamente per epoca
    model1.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
 
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model1(images)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item() * images.size(0)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)
 
    val_loss = val_loss_sum / val_total
    val_acc = 100.0 * val_correct / val_total
 
    history1["train_loss"].append(train_loss)
    history1["train_acc"].append(train_acc)
    history1["val_loss"].append(val_loss)
    history1["val_acc"].append(val_acc)
 
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train loss {train_loss:.4f}, acc {train_acc:.2f}% | "
          f"Val loss {val_loss:.4f}, acc {val_acc:.2f}%")
 
    # regola di selezione (uguale per tutti i metodi): massima accuracy sul validation
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_weights = copy.deepcopy(model1.state_dict())  # model.state_dict() restituisce un dizionario con tutti i parametri del modello
 
 
sync()
train_time1 = time.time() - t0  # include anche la valutazione sul validation a ogni epoca
model1.load_state_dict(best_weights)
 
torch.save(best_weights, "model1_full_finetuning.pt")  # checkpoint selezionato
histories["Full fine-tuning"] = history1
plot_history(history1, "Full fine-tuning", "curves_full_ft.png")
print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
print(f"Training time: {train_time1:.1f}s")
 
 
# TEST: una sola valutazione sul test del checkpoint scelto con il validation
train_pred1, y_train1 = predict_nn(model1, train_eval_loader)
test_pred1, y_test1 = predict_nn(model1, test_loader)
add_result("Full fine-tuning", y_train1, train_pred1, y_test1, test_pred1,
           best_val=f"val acc {best_val_acc:.2f}%", n_params=count_trainable(model1),
           train_time=train_time1, selected=f"epoch {best_epoch}", cm_file="confusion_matrix_full_ft.png")
show_examples(test_dataset, y_test1, test_pred1, "Full fine-tuning: correct (top) / wrong (bottom)", "examples_full_ft.png")
print("\n")
 
 
############################# Classifier head training #################################
best_val_acc = 0
best_weights = None
best_epoch = None
history2 = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
 
model2 = build_model()
for param in model2.features.parameters():
    param.requires_grad = False
# trainabili: classifier[0] = Linear(576, 1024) e classifier[3] = Linear(1024, 37)
# (classifier[1] Hardswish e classifier[2] Dropout non hanno parametri)
 
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model2.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
# lambda p: p.requires_grad restituisce True se il parametro è allenabile, filter tiene solo quelli
criterion = nn.CrossEntropyLoss()
 
 
print("CLASSIFIER HEAD TRAINING\n")
 
sync()
t0 = time.time()
for epoch in range(NUM_EPOCHS):
    model2.train()
    model2.features.eval()  # il feature extractor non va allenato (BatchNorm usa le statistiche di ImageNet)
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model2(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0)
    train_loss = running_loss / train_total
    train_acc = 100.0 * train_correct / train_total
    model2.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model2(images)
            loss = criterion(outputs, labels)
            val_loss_sum += loss.item() * images.size(0)
            val_correct += (outputs.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)
    val_loss = val_loss_sum / val_total
    val_acc = 100.0 * val_correct / val_total
 
    history2["train_loss"].append(train_loss)
    history2["train_acc"].append(train_acc)
    history2["val_loss"].append(val_loss)
    history2["val_acc"].append(val_acc)
 
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train loss {train_loss:.4f}, acc {train_acc:.2f}% | "
          f"Val loss {val_loss:.4f}, acc {val_acc:.2f}%")
 
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_weights = copy.deepcopy(model2.state_dict())
sync()
train_time2 = time.time() - t0
 
model2.load_state_dict(best_weights)
torch.save(best_weights, "model2_head_training.pt")
histories["Classifier-head training"] = history2
plot_history(history2, "Classifier-head training", "curves_head.png")
print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
print(f"Training time: {train_time2:.1f}s")
 
train_pred2, y_train2 = predict_nn(model2, train_eval_loader)
test_pred2, y_test2 = predict_nn(model2, test_loader)
add_result("Classifier-head training", y_train2, train_pred2, y_test2, test_pred2,
           best_val=f"val acc {best_val_acc:.2f}%", n_params=count_trainable(model2),
           train_time=train_time2, selected=f"epoch {best_epoch}", cm_file="confusion_matrix_head.png")
show_examples(test_dataset, y_test2, test_pred2, "Head training: correct (top) / wrong (bottom)", "examples_head.png")
print("\n")
 
 
############################# Feature extractor with Logistic Regression ################################
 
model3 = models.mobilenet_v3_small(weights=weights)
model3.classifier = nn.Identity()  # l'output diventa il vettore da 576 dopo l'avgpool
 
for param in model3.parameters():
    param.requires_grad = False
model3.eval()
model3 = model3.to(device)
 
def extract_features(model, loader):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            feats = model(images)  # estraggo features, una riga per immagine
            all_features.append(feats.cpu().numpy())  # trasporto su cpu e converto in numpy per sklearn
            all_labels.append(labels.numpy())  # non trasporto su cpu perchè non le avevo spostate su device prima
    return np.concatenate(all_features), np.concatenate(all_labels)  # concateno le feature e le etichette
 
# stesse immagini di train_set con la stessa train_transform delle reti; shuffle=False perché sklearn non dipende dall'ordine
train_aug_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
 
# il costo di training è estrazione delle feature di TRAIN (con augmentation) + fit, quindi misuro il train a parte
torch.manual_seed(SEED)  # crop e flip casuali riproducibili, indipendenti da quanto random è stato consumato prima
sync()
t0 = time.time()
X_train_aug_list, y_train_aug_list = [], []
for k in range(N_AUG_COPIES):
    # ogni passata applica crop e flip casuali diversi, come una nuova epoca delle reti
    X_k, y_k = extract_features(model3, train_aug_loader)
    X_train_aug_list.append(X_k)
    y_train_aug_list.append(y_k)
X_train_aug = np.concatenate(X_train_aug_list)  # N_AUG_COPIES * N_train righe
y_train_aug = np.concatenate(y_train_aug_list)
sync()
train_extraction_time = time.time() - t0
 
t0 = time.time()
X_train, y_train = extract_features(model3, train_eval_loader)  # train senza augmentation: serve solo per la train accuracy, come per le reti
X_val, y_val = extract_features(model3, val_loader)
X_test, y_test = extract_features(model3, test_loader)
sync()
other_extraction_time = time.time() - t0
 
np.savez("features.npz", X_train_aug=X_train_aug, y_train_aug=y_train_aug, X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
 
print("\nFEATURES EXTRACTOR + LOGISTIC REGRESSION\n")
print(f"Feature shape: train augmented {X_train_aug.shape} ({N_AUG_COPIES} copies), train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
print(f"Feature extraction time: train augmented {train_extraction_time:.1f}s, train+val+test {other_extraction_time:.1f}s")
 
# regola di selezione uguale alle reti: massima accuracy sul validation (la macro-F1 viene solo stampata)
best_val_acc_lr = 0
best_clf = None
best_C = None
 
 
for C in LOGREG_C_GRID:
    clf = Pipeline([('scaler', StandardScaler()), ("logreg", LogisticRegression(C=C, max_iter=LOGREG_MAX_ITER, random_state=SEED))])
    # per ogni colonna j delle 576 calcola media μⱼ e deviazione standard σⱼ, e trasforma ogni valore in (x − μⱼ) / σⱼ. Così ogni feature ha media 0 e deviazione 1,
    # la regressione logistica converge più velocemente sui dati standardizzati, il parametro max_iter controlla il numero massimo di iterazioni,
    # il parametro random_state assicura la riproducibilità del risultato
    # impara una matrice di pesi W (37×576) e un bias b (37). Per ogni immagine calcola i 37 punteggi W·x + b, li trasforma in probabilità con la softmax e prende la classe più probabile
    t0 = time.time()
    clf.fit(X_train_aug, y_train_aug)  # addestro il classificatore sulle features aumentate (lo scaler è fittato solo sul train)
    fit_time = time.time() - t0
    val_predict = clf.predict(X_val)  # faccio predizioni sul validation set
    val_accuracy = accuracy_score(y_val, val_predict)  # calcolo l'accuratezza
    val_f1 = f1_score(y_val, val_predict, average="macro")  # calcolo il macro-f1 score
    print(f"C={C}: val acc {val_accuracy*100:.2f}%, val macro-F1 {val_f1:.4f}, fit {fit_time:.1f}s")
 
    if val_accuracy > best_val_acc_lr:
        best_val_acc_lr = val_accuracy
        best_clf = clf
        best_C = C
        best_fit_time = fit_time
 
print(f"Best C: {best_C}, Best val acc: {best_val_acc_lr*100:.2f}%")
 
n_params = best_clf["logreg"].coef_.size + best_clf["logreg"].intercept_.size
print(f"Number of trainable parameters: {n_params}\n")
joblib.dump(best_clf, "model3_logreg.joblib")  # salvo scaler + classificatore selezionati
 
train_pred3 = best_clf.predict(X_train)
test_pred3 = best_clf.predict(X_test)
add_result("Features + Logistic Regression", y_train, train_pred3, y_test, test_pred3,
           best_val=f"val acc {best_val_acc_lr*100:.2f}%", n_params=n_params,
           train_time=train_extraction_time + best_fit_time, selected=f"C={best_C}",
           cm_file="confusion_matrix_logreg.png")
show_examples(test_dataset, y_test, test_pred3, "Features + LogReg: correct (top) / wrong (bottom)", "examples_logreg.png")
 
 
############################### Feature extractor with linear SVM ################################
 
 
best_val_acc_svm = 0
best_svm = None
best_C_svm = None
 
print("\nFEATURES EXTRACTOR + LINEAR SVM \n")
 
for C in SVM_C_GRID:
    svm = Pipeline([('scaler', StandardScaler()), ("svm", LinearSVC(C=C, max_iter=SVM_MAX_ITER, random_state=SEED))])
    t0 = time.time()
    svm.fit(X_train_aug, y_train_aug)
    fit_time = time.time() - t0
    val_predict = svm.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predict)
    val_f1 = f1_score(y_val, val_predict, average="macro")
    print(f"C={C}: val acc {val_accuracy*100:.2f}%, val macro-F1 {val_f1:.4f}, fit {fit_time:.1f}s")
 
    if val_accuracy > best_val_acc_svm:
        best_val_acc_svm = val_accuracy
        best_svm = svm
        best_C_svm = C
        best_fit_time_svm = fit_time
 
print(f"Best C: {best_C_svm}, Best val acc: {best_val_acc_svm*100:.2f}%")
 
n_params_svm = best_svm["svm"].coef_.size + best_svm["svm"].intercept_.size
print(f"Number of trainable parameters: {n_params_svm}\n")
joblib.dump(best_svm, "model4_linear_svm.joblib")
 
train_pred4 = best_svm.predict(X_train)
test_pred4 = best_svm.predict(X_test)
add_result("Features + Linear SVM", y_train, train_pred4, y_test, test_pred4,
           best_val=f"val acc {best_val_acc_svm*100:.2f}%", n_params=n_params_svm,
           train_time=train_extraction_time + best_fit_time_svm, selected=f"C={best_C_svm}",
           cm_file="confusion_matrix_svm.png")
show_examples(test_dataset, y_test, test_pred4, "Features + SVM: correct (top) / wrong (bottom)", "examples_svm.png")
 
 
############################### tabella finale dei risultati ################################
 
print("\nRESULTS TABLE\n")
header = (f"{'Method':<32}{'Selected':>12}{'Test acc':>10}{'95% CI':>16}{'Macro-F1':>10}"
          f"{'Train acc':>11}{'Gap':>8}{'Params':>12}{'Time (s)':>10}")
print(header)
print("-" * len(header))
for r in results:
    ci = f"{r['test_acc_ci95'][0]:.2f}-{r['test_acc_ci95'][1]:.2f}"
    print(f"{r['method']:<32}{r['selected']:>12}{r['test_acc']:>9.2f}%{ci:>16}{r['test_macro_f1']:>10.4f}"
          f"{r['train_acc']:>10.2f}%{r['train_test_gap']:>8.2f}{r['trainable_params']:>12,}{r['train_time_s']:>10.1f}")
 
# salvo metriche e configurazione per il repository
run_info = {
    "versions": {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "sklearn": sklearn.__version__,
    },
    "device": str(device),
    "seed": SEED,
    "cudnn_deterministic": True,
    "deterministic_algorithms": "True (warn_only)",
    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "pretrained_weights": str(weights),
    "class_imbalance_ratio": {k: round(v, 3) for k, v in imbalance.items()},
    "hyperparameters": {
        "neural_networks": {"batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs": NUM_EPOCHS,
                            "optimizer": "AdamW", "loss": "CrossEntropyLoss",
                            "selection_rule": "epoch with max validation accuracy"},
        "logistic_regression": {"C_grid": LOGREG_C_GRID, "max_iter": LOGREG_MAX_ITER, "scaler": "StandardScaler (fit on train)",
                                "train_augmentation": f"{N_AUG_COPIES} augmented copies of the training set (train_transform)",
                                "selection_rule": "C with max validation accuracy"},
        "linear_svm": {"C_grid": SVM_C_GRID, "max_iter": SVM_MAX_ITER, "scaler": "StandardScaler (fit on train)",
                       "train_augmentation": f"{N_AUG_COPIES} augmented copies of the training set (train_transform)",
                       "selection_rule": "C with max validation accuracy"},
        "val_fraction": VAL_FRACTION,
        "bootstrap_samples": N_BOOTSTRAP,
    },
    "histories": histories,
    "results": results,
}
with open("results.json", "w") as f:
    json.dump(run_info, f, indent=2)
 
with open("results.csv", "w") as f:
    f.write("method,selected,test_acc,test_acc_ci_low,test_acc_ci_high,test_macro_f1,train_acc,train_test_gap,trainable_params,train_time_s\n")
    for r in results:
        f.write(f"{r['method']},{r['selected']},{r['test_acc']},{r['test_acc_ci95'][0]},{r['test_acc_ci95'][1]},"
                f"{r['test_macro_f1']},{r['train_acc']},{r['train_test_gap']},{r['trainable_params']},{r['train_time_s']}\n")
