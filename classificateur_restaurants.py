import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Lire les avis et leurs étiquettes depuis le fichier CSV
with open("avis_restaurants.csv", encoding="utf-8", newline="") as fichier:
    lignes = list(csv.DictReader(fichier, delimiter=";"))

avis = [ligne["avis"] for ligne in lignes]
sentiments = [ligne["sentiment"] for ligne in lignes]

# Entraîner le modèle avec les avis du CSV
modele = make_pipeline(CountVectorizer(ngram_range=(1, 2)), MultinomialNB())
avis_train, avis_test, sentiments_train, sentiments_test = train_test_split(
    avis, sentiments, test_size=0.25, random_state=42, stratify=sentiments
)

modele.fit(avis_train, sentiments_train)

avis_predits = modele.predict(avis_test)

for texte, attendu, predit in zip(avis_test, sentiments_test, avis_predits):
    if attendu != predit:
        print(f"Erreur : {texte} | attendu : {attendu} | prédit : {predit}")

taux = modele.score(avis_test, sentiments_test)
print(f"Taux de bonnes réponses : {taux:.0%}")
modele.fit(avis, sentiments)

print(f"Modèle entraîné avec {len(avis)} avis.")

# Tester plusieurs avis
while True:
    nouvel_avis = input("Écris un avis (ou 'quitter' pour arrêter) : ")

    if nouvel_avis.lower() == "quitter":
        break

    prediction = modele.predict([nouvel_avis])[0]
    print(f"Sentiment prédit : {prediction}")
