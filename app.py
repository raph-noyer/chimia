import streamlit as st
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.Chem.Draw import SimilarityMaps

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Predicteur BHE", page_icon="🧠")

# --- TITRE ET INTRO ---
st.title("🧠 Neuro-Druggability Predictor")
st.markdown("""
Cette application utilise une **Intelligence Artificielle (Random Forest)** pour prédire si une molécule 
peut traverser la **barrière hémato-encéphalique (BHE)**.
*Modèle entraîné sur le dataset BBBP (84% accuracy).*
""")

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource # On met le modèle en cache pour que ça aille vite
def load_model():
    # Assure-toi que le fichier .pkl est au même endroit que ce script
    try:
        with open('model_qsar.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("❌ Fichier 'model_qsar.pkl' introuvable. Veuillez l'uploader.")
        return None

model = load_model()

# --- FONCTIONS UTILITAIRES (Celles de ton Colab) ---
# AVANT (Ce qui plante) :
# def get_fingerprint(mol):

# APRÈS (La correction) :
def get_fingerprint(mol, atomId=-1):
    # RDKit a besoin de cet argument 'atomId', même si on ne l'utilise pas ici.
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def get_proba(fp):
    # L'IA attend une liste de vecteurs, on l'enveloppe donc dans []
    return model.predict_proba([fp])[0][1]

# --- INTERFACE UTILISATEUR ---
st.sidebar.header("Entrée Moléculaire")
smiles_input = st.sidebar.text_input(
    "Entrez un code SMILES :",
    value="CN(C)C(=O)O" # Une molécule par défaut pour l'exemple
)

if st.sidebar.button("Lancer l'analyse"):
    if model is not None:
        mol = Chem.MolFromSmiles(smiles_input)
        
        if mol:
            # 1. Prédiction
            fp = get_fingerprint(mol)
            proba = model.predict_proba([fp])[0][1]
            class_pred = "Perméable (Passe dans le cerveau)" if proba > 0.5 else "Non-Perméable"
            color = "green" if proba > 0.5 else "red"
            
            # Affichage du score
            st.metric(label="Prédiction IA", value=f"{proba:.1%} - {class_pred}")
            
            # 2. Visualisation (Similarity Map)
            st.subheader("🔍 Analyse d'explicabilité (XAI)")
            st.write("Les zones vertes favorisent le passage, les zones rouges le bloquent.")
            
            # Création du dessin (Technique SVG robuste)
            d = MolDraw2DSVG(600, 500)
            SimilarityMaps.GetSimilarityMapForModel(
                mol, get_fingerprint, get_proba, draw2d=d, contourLines=10
            )
            d.FinishDrawing()
            svg = d.GetDrawingText()
            
            # Affichage du SVG dans Streamlit
            st.image(svg, use_column_width=False) # Streamlit gère le SVG directement maintenant
            
        else:
            st.error("SMILES invalide. Vérifiez votre syntaxe.")

# --- FOOTER ---
st.markdown("---")
st.caption("Projet réalisé avec Python, RDKit et Streamlit, par Raphael Noyer.")
