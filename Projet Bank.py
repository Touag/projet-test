#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from scipy.stats import pearsonr, chi2_contingency
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, LabelEncoder
import statsmodels.api
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from PIL import Image


# 1. Accueil : Expliquer le projet et le but de l'application
# 2. Présentation du jeu de donnée
# 3. Analyse exploratoire des données : Afficher les graphiques afin d'expliquer les relations entre les variables
# 4. Sélection des paramètres pour les prédictions : modèles + hyperparamètres + résultats + explications
# 5. Interface des prédictions : Fournir un formulaire pour entrer les valeurs souhaitées et voir la prédiction selon le modèle choisit

# In[2]:



pages = ["Introduction", "Présentation du jeu de donnée", "Analyse exploratoire", "Arbitrage & preprocessing", "Modèles & Hyperparamètres", "Votre prédiction"]
df = pd.read_csv("Bank.csv")
page = st.sidebar.radio("Les étapes :", pages)

if page == pages[0]:
    st.title("**Prédiction du succès d’une campagne de Marketing d’une banque**")
    st.write("   ")
    st.write("   ")
    st.write('Bienvenue sur notre application de prédiction pour la souscription à un produit financier.')
    st.write("   ")
    st.write("**_Durant notre formation de Data Analyst chez DataScientest, nous avons eu l'opportunité de travailler sur un projet professionnel passionnant : prédire la souscription à un produit financier d'un client à partir des données collectées sur lui ainsi que celles de la précédente campagne marketing. Nous avons mis en œuvre des techniques de classification avancées pour construire un modèle performant capable de prendre en compte toutes les informations pertinentes. Ce projet a été une expérience inestimable pour notre formation en tant que futurs professionnels de l'analyse de données._**")


# In[3]:


if page == pages[1]:
    st.title("Première analyse et interprétation des variables")
    st.write("   ")
    st.write("L'enjeu était de découvrir le dataset, l'explorer, le préparer et créer un modèle prédictif pour anticiper et maximiser les résultats d'une prochaine campagne.")
    st.write("Voici un premier aperçu du jeu de donnée :", df.head(5))
    st.write("**Dimensions du dataset** : Celui-ci est constitué de", df.shape[0],"lignes et de", df.shape[1],"colonnes.")
    st.write("**Nombre de doublons** présents dans le jeu de donnée :", df.duplicated().sum(),".")
    st.write("**Nombre de NANs** présents :", df.isna().any().sum(),".")
    st.write("  ")
    
    st.header("**Identification et interprétation des variables** :")
    st.write("1. **age** : _l'âge du client en années (variable numérique)_")
    st.write("2. **job** : _l'activité professionnelle du client (variable catégorielle)_")
    st.write("3. **marital** : _l'état matrimonial du client (variable catégorielle)_")
    st.write("4. **education** : _le niveau d'études du client (variable catégorielle)_")
    st.write("5. **default** : _indique si le client a déjà été en situation de défaut de paiement (variable catégorielle)_")
    st.write("6. **balance** : _le solde du compte du client (variable numérique)_")
    st.write("7. **housing** : _indique si le client possède un prêt immobilier (variable catégorielle)_")
    st.write("8. **loan** : _indique si le client possède un prêt personnel en cours (variable catégorielle)_")
    st.write("9. **contact** : _le moyen de communication utilisé pour contacter le client (variable catégorielle)_")
    st.write("10. **day** : _le jour du mois où le client a été contacté pour la dernière fois (variable numérique)_")
    st.write("11. **month** : _le mois de l'année où le client a été contacté pour la dernière fois (variable catégorielle)_")
    st.write("12. **duration** : _la durée en secondes du dernier échange avec le client lors de la précédente campagne (variable numérique)_")
    st.write("13. **campaign** : _le nombre de contacts effectués pendant cette campagne pour ce client (variable numérique)_")
    st.write("14. **pdays** : _le nombre de jours qui se sont écoulés depuis le dernier contact avec le client lors d'une campagne antérieure (variable numérique)_")
    st.write("15. **previous** : _le nombre de contacts effectués lors de la campagne antérieure pour ce client (variable numérique)_")
    st.write("16. **poutcome** : _le résultat de la campagne marketing précédente (variable catégorielle)_")
    st.write("17. **deposit** : _indique si le client a effectué un dépôt à terme lors de cette campagne Yes / No (variable catégorielle)_")
    st.write("   ")
    st.write("   ")
    st.write("   ")
    st.write("**_Notre objectif est de comprendre les facteurs qui influencent la décision des clients de souscrire à un produit bancaire spécifique et de prédire s'ils souscriront à l'avenir à l'aide d'un modèle de machine learning. La variable cible, nommée 'deposit', indique si un client a effectué un dépôt sur le contrat terme._**") 


# In[4]:


names_columns = df.columns.tolist()
num_vars = df.select_dtypes(include=["int64", "float64"])

if page == pages[2]:
    st.title("Sélectionner la variable à explorer :")
    option = st.selectbox("", names_columns)

# age
    if option == "age" :
        st.write("Présentation de la variable : _l'âge du client en années (variable numérique)_")
        ## Histogramme avec courbe de la densité 
        fig_age = sns.displot(num_vars.age, kde=True)
        plt.title("Distribution de la variable âge")
        st.pyplot(fig_age)
        st.write("La distribution de cette variable se concentre entre 25 ans et 50 ans, avec une moyenne à 41 ans et une médiane à 39 ans. On note que 75% des modalités se situent sous les 49 ans.")
        
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Analyse de la variance ANOVA :")
        result_age = statsmodels.formula.api.ols("age ~ deposit", data=df).fit()
        table_age = statsmodels.api.stats.anova_lm(result_age)
        table_age
        
        # Créer deux subplots
        fig_sub, (ax1, ax2) = plt.subplots(1, 2)

        # Représentation graphique
        sns.violinplot(x="deposit", y="age", data=df, ax=ax1)
        ax1.set_title("Répartition de l'âge en fonction de la variable deposit", fontsize=8)

        # Focus sur la tranche d'âge entre 25 et 60 ans
        df_age = df[(df["age"] > 25) & (df["age"]<60)]
        sns.violinplot(x="deposit", y="age", data=df_age, ax=ax2)
        ax2.set_title("Zoom de 25 ans à 50 ans", fontsize=8)

        # Afficher les subplots dans Streamlit
        st.write("Visualisation par graphique :")
        st.pyplot(fig_sub)
        
# job        
    if option == "job":
        st.write("Présentation de la variable : _l'activité professionnelle du client (variable catégorielle)_")

    # Distribution de la variable job
        fig_job, ax = plt.subplots()
        df.job.value_counts().plot(kind='bar')
        ax.set_title("Distribution de la variable job")
        st.pyplot(fig_job)
    
        st.write("Nous notons la présence de la modalité 'unknown' qui pourrait s'apparenter à des NaNs. Nous allons voir sur la façon de les traiter.")
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Test de Chi2 :")
        table = pd.crosstab(df['deposit'], df['job'])
        chi2, pval, dof, expected = chi2_contingency(table)
        table_job = pd.DataFrame({"Statistique du test" : [chi2],
                                 "P-value" : [pval],
                                 "Degré de liberté" : [dof]})
        st.write(table_job)
    # Barplot de la variable job en fonction de la variable cible
        fig_job_target, ax = plt.subplots()
        sns.countplot(x=df['job'], hue=df['deposit'], order=df['job'].value_counts().index)
        ax.set_title("Répartition de la variable job en fonction de la variable cible")
        ax.legend(loc='upper right')
        plt.xticks(rotation=45) # Rotation de 90 degrés pour les étiquettes de l'axe X
        st.write("Visualisation par graphique :")
        st.pyplot(fig_job_target)

# marital        
    if option == "marital":
        st.write("Présentation de la variable : _l'état matrimonial du client (variable catégorielle)_")

    # Distribution de la variable marital
        fig_marital, ax = plt.subplots()
        df.marital.value_counts().plot(kind='bar')
        ax.set_title("Distribution de la variable marital")
        st.pyplot(fig_marital)
        st.write("La donnée semble propre avec l'absence de nan ou de valeur anormale.")
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Test de Chi2 :")
        table_marital = pd.crosstab(df['deposit'], df['marital'])
        chi2, pval, dof, expected = chi2_contingency(table_marital)
        df_marital = pd.DataFrame({"Statistique du test" : [chi2],
                                 "P-value" : [pval],
                                 "Degré de liberté" : [dof]})
        st.write(df_marital)
    # Countplot de la variable marital en fonction de la variable cible
        fig_marital_target, ax = plt.subplots()
        sns.countplot(x=df['marital'], hue=df['deposit'], palette='Set2', order=df['marital'].value_counts().index)
        ax.set_title("Répartition de la variable marital en fonction de la variable cible")
        ax.legend(loc='upper right')
        plt.xticks(rotation=45) # Rotation de 45 degrés pour les étiquettes de l'axe X
        st.write("Visualisation par graphique :")
        st.pyplot(fig_marital_target)

# education
    if option == "education":
        st.write("Présentation de la variable : _le niveau d'études du client (variable catégorielle)_")

    # Distribution de la variable education
        st.write(df.education.value_counts())
        st.write("Nous observons la présence d'une modalité 'unknown'.")
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Test de Chi2 :")        
        table_education = pd.crosstab(df['deposit'], df['education'])
        chi2, pval, dof, expected = chi2_contingency(table_education)
        df_education = pd.DataFrame({"Statistique du test" : [chi2],
                                 "P-value" : [pval],
                                 "Degré de liberté" : [dof]})
        st.write(df_education)

    # Création d'un countplot pour la visualisation de la répartition
        fig_education, ax = plt.subplots()
        sns.countplot(x=df['education'], hue=df['deposit'], order=df['education'].value_counts().index)
        ax.set_title("Répartition du niveau d'étude")
        ax.legend(loc='upper right')
        st.write("Visualisation par graphique :")
        st.pyplot(fig_education)

#loan
    if option == "loan":
        st.write("Présentation de la variable : _indique si le client possède un prêt personnel en cours (variable catégorielle)_")

    # Distribution de la variable loan
        st.write(df.loan.value_counts())
        st.write("Cette variable est déséquilibrée, cependant on remarque clairement une tendance dans la modalité 'yes' par rapport à la variable cible. De ce fait, nous conservons cette donnée, elle sera encoder via le LabelEncoder ou OHE en vue de son utilisation dans le machine learning. Les deux encodages sont possibles.")
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Test de Chi2 :")
        table_loan = pd.crosstab(df['deposit'], df['loan'])
        chi2, pval, dof, expected = chi2_contingency(table_loan)
        df_loan = pd.DataFrame({"Statistique du test" : [chi2],
                                 "P-value" : [pval],
                                 "Degré de liberté" : [dof]})
        st.write(df_loan)
    # Création d'un countplot pour visualiser la répartition
        fig_loan, ax = plt.subplots()
        sns.countplot(x=df['loan'], hue=df['deposit'], order=df['loan'].value_counts().index)
        ax.set_title("Présence ou absence de prêt en fonction du deposit")
        ax.legend(loc='upper right')
        st.write("Visualisation par graphique :")
        st.pyplot(fig_loan)

#housing
    if option == "housing" :
        st.write("Présentation de la variable : _indique si le client possède un prêt immobilier (variable catégorielle)_")
    # Vérification des modalités
        st.write("Distribution de la variable :")
        st.write(df.housing.value_counts())
        st.write("La distribution de cette variable montre qu'il y a plus de clients ayant un crédit immobilier que de clients n'en ayant pas.")
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Test de Chi2 :")
        table_housing = pd.crosstab(df['deposit'], df['housing'])
        chi2, pval, dof, expected = chi2_contingency(table_housing)
        df_housing = pd.DataFrame({"Statistique du test" : [chi2],
                                 "P-value" : [pval],
                                 "Degré de liberté" : [dof]})
        st.write(df_housing)
    # Création d'un countplot pour visualiser la répartition en fonction de la variable cible
        fig, ax = plt.subplots()        
        sns.countplot(x=df['housing'], hue=df['deposit'])
        plt.title("Présence ou absence d'un prêt immobilier avec sa répartition sur deposit", fontsize=9)
        plt.legend()
        st.write("Visualisation par graphique :")
        st.pyplot(fig)
        st.write("Très clairement nous avons une tendance visuelle sur le graphique. Les clients ayant contracté un crédit immobilier sont plus susceptibles de ne pas faire un dépôt long terme et inversement pour les clients sans crédit immobilier. On peut parler de variables anti-corrélées.")
        st.write("On appliquera un préprocessing de type OHE ou Labelencoder en vue de son utilisation pour le machine learning.")

# default        
    if option == "default":
        st.write("Présentation de la variable : _indique si le client a déjà été en situation de défaut de paiement (variable catégorielle)_")
    
    # Vérification des modalités
        st.write("Distribution de la variable :")
        st.write(df.default.value_counts())
        st.write("On note un déséquilibre très important.")
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Test de Chi2 :")
        table_default = pd.crosstab(df['deposit'], df['default'])
        chi2, pval, dof, expected = chi2_contingency(table_default)
        df_default = pd.DataFrame({"Statistique du test" : [chi2],
                                 "P-value" : [pval],
                                 "Degré de liberté" : [dof]})
        st.write(df_default)
    # Création d'un countplot pour visualiser la répartition en fonction de la variable cible
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.countplot(x=df['default'], hue=df['deposit'], ax=ax)
        ax.set_title("Présence ou absence de défaut de paiement avec sa répartition sur deposit", fontsize=9);
        ax.legend()
        st.write("Visualisation par graphique :")
        st.pyplot(fig)    
        st.write("Au vu de son déséquilibre très important et de sa faible importance sur la variable cible, nous avons décidé de supprimer cette variable du dataset. Son utilisation pourrait fausser le résultat lors des tests en machine learning.")

# poutcome        
    if option == "poutcome":
        st.write("Présentation de la variable : _le résultat de la campagne marketing précédente (variable catégorielle)_")
        
        # Vérification des modalités
        st.write("Distribution de la variable :")
        st.write(df.poutcome.value_counts())
        st.write("La distribution de cette variable montre qu'il y a une majorité de clients pour lesquels nous n'avons pas d'informations sur la campagne précédente (unknown).")
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Test de Chi2 :")
        table_poutcome = pd.crosstab(df['deposit'], df['poutcome'])
        chi2, pval, dof, expected = chi2_contingency(table_poutcome)
        df_poutcome = pd.DataFrame({"Statistique du test" : [chi2],
                                 "P-value" : [pval],
                                 "Degré de liberté" : [dof]})
        st.write(df_poutcome)
        # Création d'un countplot pour visualiser la répartition en fonction de la variable cible
        fig, ax = plt.subplots()
        sns.countplot(x='poutcome', hue='deposit', data=df, ax=ax)
        plt.title("Influence de la campagne précédente sur deposit")
        st.write("Visualisation par graphique :")
        st.pyplot(fig)
        
        st.write("Lorsque nous visualisons cette variable avec les variables pdays et previous, nous établissons un lien entre les unknowns de cette variable, la modalité -1 de 'pdays' et la modalité 0 de 'previous'.")
        st.write("Nous émettons l'hypothèse que toutes les personnes ayant ces modalités sur ces 3 variables sont en faite des personnes qui n'ont pas participé à la campagne précédente et qu'il s'agit potentiellement de 'nouveaux clients'.")
        st.write("De ce fait, la modalité 'unknown' est une information en soit. Nous décidons donc de la conserver en l'état et elle sera transformée lors du preprocessing via l'OHE.")

# month        
    if option == "month":
        st.write("Présentation de la variable : _le mois de l'année où le client a été contacté pour la dernière fois (variable catégorielle)_")

        # Fréquence des modalités de la variable
        st.write("Distribution de la variable :")
        st.write(df.month.value_counts())
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Test de Chi2 :")
        table_month = pd.crosstab(df['deposit'], df['month'])
        chi2, pval, dof, expected = chi2_contingency(table_month)
        df_month = pd.DataFrame({"Statistique du test" : [chi2],
                                 "P-value" : [pval],
                                 "Degré de liberté" : [dof]})
        st.write(df_month)
        # Création d'un countplot en fonction du deposit
        fig, ax = plt.subplots()
        sns.countplot(x='month', hue='deposit', data=df)
        plt.title("Répartition du deposit en fonction du mois")

        # Afficher le graphique dans Streamlit
        st.write("Visualisation par graphique :")
        st.pyplot(fig)

        st.write("Il apparaît clairement que le volume du mois de mai est bien supérieur aux autres mois. Plusieurs questions se posent : est-ce que la campagne a démarré au mois de mai, justifiant un engagement important des équipes ? Est-ce que la campagne a duré plus d'un an et s'est terminée au mois de mai de l'année suivante ? Avec deux mois de mai comptabilisés, cela expliquerait l'écart. ")

# contact 
    if option == "contact":
        st.write("Présentation de la variable : _le moyen de communication utilisé pour contacter le client (variable catégorielle)_")
    
    # Afficher la distribution des modalités de la variable
        st.write("Fréquence des modalités de la variable :")
        st.write(df.contact.value_counts())
        
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Test de Chi2 :")
        table_contact = pd.crosstab(df['deposit'], df['contact'])
        chi2, pval, dof, expected = chi2_contingency(table_contact)
        df_contact = pd.DataFrame({"Statistique du test" : [chi2],
                                 "P-value" : [pval],
                                 "Degré de liberté" : [dof]})
        st.write(df_contact)
    # Créer un countplot pour visualiser la répartition en fonction de la variable cible
        fig, ax = plt.subplots()
        colors = sns.color_palette("Set2")
        sns.countplot(x=df['contact'], hue=df['deposit'], palette = colors, order = df['contact'].value_counts().index)
        plt.title("Les moyens de communications avec le client")
        plt.legend()
        st.write("Visualisation par graphique :")
        st.pyplot(fig)
    
        st.write("Nous émettons l'hypothèse que la modalité 'unknown' signifie que nous ne possédons pas l'information du numéro de téléphone du client. Nous avons évoqué la possibilité de synthétiser cette variable en 'Yes / No' pour préciser si nous avions l'information du numéro de téléphone pour contacter le client. Finalement, elle sera conservée en l'état en vue de sa transformation lors du preprocessing via le 'OneHotEncoding'.")

# balance        
    if option == "balance" :
        st.write("Présentation de la variable : _le solde du compte du client (variable numérique)_")
        # Visualisation par graphique : Boxplot
        fig_balance, ax = plt.subplots()
        ax.boxplot(df.balance)
        ax.set_title("Répartition de la variable balance")
        st.pyplot(fig_balance)
        st.write("Médiane faible, avec une grosse dispersion")
        
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Analyse de la variance ANOVA :")
        result_balance = statsmodels.formula.api.ols("balance ~ deposit", data=df).fit()
        table_balance = statsmodels.api.stats.anova_lm(result_balance)
        table_balance
        
        # Créer deux subplots
        fig_sub, (ax1, ax2) = plt.subplots(1, 2)

        # Représentation graphique
        sns.violinplot(x="deposit", y="age", data=df, ax=ax1)
        ax1.set_title("Répartition de l'âge en fonction de la variable deposit", fontsize=8)

        # Focus sur la tranche d'âge entre 25 et 60 ans
        df_age = df[(df["age"] > 25) & (df["age"]<60)]
        sns.violinplot(x="deposit", y="age", data=df_age, ax=ax2)
        ax2.set_title("Zoom de 25 à 50 ans", fontsize=8)

        # Afficher les subplots dans Streamlit
        st.write("Visualisation par graphique :")
        st.pyplot(fig_sub)
        

#day       
    if option == "day" :
        st.write("Présentation de la variable : _le jour du mois où le client a été contacté pour la dernière fois (variable numérique)_")
        
        st.write("Les indicateurs de positions sur cette variable :")
        st.write(num_vars.day.describe())
        #visualisation par graphique: barplot
        fig_day, ax = plt.subplots()
        sns.countplot(x=num_vars.day);
        plt.xticks([0,4,9,14,19,24,30]);
        plt.title("Nb de contact en fonction du jour")
    
        # Display the plot in Streamlit
        st.pyplot(fig_day)
                
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Analyse de la variance ANOVA :")
        result_day = statsmodels.formula.api.ols("day ~ deposit", data=df).fit()
        table_day = statsmodels.api.stats.anova_lm(result_day)
        table_day
        st.write("Cette variable aurait pu être intéressante si nous connaissions l’année, cela nous aurait permis d’étudier la saisonnalité en fonction du jour de la semaine ou du mois pour évaluer les performances.")

#duration
    if option == "duration" :
        st.write("Présentation de la variable : _la durée en secondes du dernier échange avec le client lors de la précédente campagne (variable numérique)_")
        
        st.write("Les indicateurs de positions sur cette variable")
        st.write(num_vars.duration.describe())
        #visualisation par barplot
        fig_duration = sns.displot(num_vars.duration, bins=30, color='blue')
        plt.title("Distribution de la variable âge")
        st.pyplot(fig_duration)
        st.write("La moyenne se situe à 371 secondes avec une médiane à 255 secondes. On note également que 75% des appels n’excèdent pas 496 secondes soit 8.26 minutes environ.")
      
        #analyse de la variance avec ANOVA avec la variable cible
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Analyse de la variance ANOVA :")
        result_duration = statsmodels.formula.api.ols("duration ~ deposit", data=df).fit()
        table_duration = statsmodels.api.stats.anova_lm(result_duration)
        table_duration
        st.write("Il s'agit de la variable ayant le plus fort impact sur la variable cible.")
       
        # Représentation graphique
        fig_sub = sns.FacetGrid(df, col="deposit")
        fig_sub.map(plt.hist, "duration");
        st.write("Visualisation par graphique :")
        st.pyplot(fig_sub)
        st.write("On note sans vraiment de surprise qu’un appel court diminue les chances de dépôt du client alors qu’un client accroché où nous avons la possibilité de dérouler l’argumentaire, augmente grandement les chances de réussite. Il semblerait que cette variable aura une grande importance lorsque nous passerons à la machine learning. Cependant cette variable n’est connue que lorsque l’action du contact a été réalisée. Nous essayerons de comprendre la manière dont la machine learning évaluera cette variable surtout au vu de son influence beaucoup plus importante que les autres variables. Cette variable sera transformée via une normalisation en vue de son utilisation.")


#campaign
    if option == "campaign" :
        st.write("Présentation de la variable : _le nombre de contacts effectués pendant cette campagne pour ce client (variable numérique)_")
        st.write("Les indicateurs de positions sur cette variable :")
        st.write(num_vars.campaign.describe())
        
        ## visualisation par graphique boxplot 
        fig_campaign, ax = plt.subplots()
        ax.boxplot(df.campaign);
        ax.set_title("Répartition de la variable campaign");
        st.pyplot(fig_campaign)
        st.write("La boite à moustache montre de façon nette une grande disparité avec une médiane très faible située à 2. Si la majorité des clients n’ont été contactés que quelques fois, 210 clients soit 1.88%, ont été contactés plus d’une 10aine de fois.")
        ## Analyse de la variance ANOVA avec la variable cible
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Analyse de la variance ANOVA :")
        result_campaign = statsmodels.formula.api.ols("campaign ~ deposit", data=df).fit()
        table_campaign = statsmodels.api.stats.anova_lm(result_campaign)
        table_campaign
        
        # Créer deux subplots
        fig_campaign, (ax1, ax2) = plt.subplots(1, 2)

        # Représentation graphique
        sns.violinplot(x="deposit", y="campaign", data=df, ax=ax1)
        ax1.set_title("Représentation du nombre de contact en fonction du deposit", fontsize=6)

        # Focus sur la tranche d'âge entre 25 et 60 ans
        df_campaign = df[df["campaign"]>10]
        sns.violinplot(x="deposit", y="age", data=df_campaign, ax=ax2)
        ax2.set_title("Zoom sur les clients ayant été contacté plus de 10 fois", fontsize=6)

        # Afficher les subplots dans Streamlit
        st.write("Visualisation par graphique :")
        st.pyplot(fig_campaign)    
        st.write("On observe qu’il suffit de quelques contacts pour augmenter le taux de réussite et cela se confirme au-delà de 10 contacts. Cependant les chances diminuent grandement à partir d’une vingtaine de contacts. Cette variable subira une standardisation.")

#pdays
    if option == "pdays":
        st.write("Présentation de la variable : _le nombre de jours qui se sont écoulés depuis le dernier contact avec le client lors d'une campagne antérieure (variable numérique)_")
        st.write("Les indicateurs de positions sur cette variable :")                       
        st.write(num_vars.pdays.describe())
        # Afficher les indicateurs de positions sur cette variable
        df_pdays_min = df[df["pdays"] == -1]
        st.write("On observe que",round(len(df_pdays_min)/len(df)*100,2),"% ont la modalité -1.")
        st.write("Si nous mettons en lien avec la variable contact, il s'agit de tous les clients qui n'ont pas été contactés lors de la précédente campagne.")
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Analyse de la variance ANOVA :")
        #analyse ANOVA
        result_pdays = statsmodels.formula.api.ols("pdays ~ deposit", data=df).fit()
        table_pdays = statsmodels.api.stats.anova_lm(result_pdays)
        table_pdays
        st.write("Les modalités de deposit :")
        st.write(df.deposit.value_counts())
        st.write("Fréquence des modalités de la variable en fonction de deposit :")
        st.write(df_pdays_min.deposit.value_counts())
        st.write("Alors que la variable deposit semble equilibrée, cet équilibre ne se retrouve pas pour les clients n'ayant pas eu de contact lors de la précédente campagne.")
         # Créer un countplot pour visualiser la répartition en fonction de la variable cible
        fig_pdays, ax = plt.subplots()
        sns.countplot(x="pdays", hue="deposit", data=df_pdays_min);
        plt.title("Répartition des clients n'ayant pas eu de contact lors de la dernière campagne par rapport au deposit");
        plt.legend()
        st.write("Visualisation par graphique :")
        st.pyplot(fig_pdays)
         
        # Créer deux subplots
        fig_pdays, (ax1, ax2) = plt.subplots(1, 2)

        # Représentation graphique
        sns.violinplot(x="deposit", y="pdays", data=df, ax=ax1)
        ax1.set_title("Répartition en fonction du deposit", fontsize=8)

        #zoom sur les clients n'ayant pas eu de contact depuis 20 jours
        df_pdays = df[df["pdays"]>20]
        sns.violinplot(x="deposit", y="pdays", data=df_pdays, ax=ax2)
        ax2.set_title("Zoom sur les clients n'ayant pas eu de contact depuis plus de 20 jours", fontsize=6)

        # Afficher les subplots dans Streamlit
        st.write("Visualisation par graphique :")
        st.pyplot(fig_pdays)  

        st.write("Il semblerait qu'un temps de réflexion de plus de 20 jours soit à l'avantage d'un dépôt sur la nouvelle campagne.")
         
#previous
    if option == "previous":
        st.write("Présentation de la variable : _le nombre de contacts effectués lors de la campagne antérieure pour ce client (variable numérique)_")    
        
        st.write("Les indicateurs de positions sur cette variable :")                       
        st.write(num_vars.previous.describe())
        
        #visualisation par boxplot
        fig_previous, ax = plt.subplots()
        ax.boxplot(df.previous);
        ax.set_title("Nb de contacts de la campagne précédente")
        st.write("Visualisation par graphique :")
        st.pyplot(fig_previous)
        
        st.header("**Étudions la relation de cette variable avec la variable cible deposit**")
        st.write("  ")
        st.write("Analyse de la variance ANOVA :")        
        result_previous = statsmodels.formula.api.ols("previous ~ deposit", data=df).fit()
        table_previous = statsmodels.api.stats.anova_lm(result_previous) 
        table_previous             
        
        #graphie de boxplot 
        fig_previous_1 = sns.catplot(x="deposit", y="previous", kind="violin", data=df);
        plt.title("Répartition du nb de contact de la campagne précédente en fonction du deposit de cette campagne")
        st.write("Visualisation par graphique :")
        st.pyplot(fig_previous_1)
        
        #focus sur les personnes n'ayant pas été contacté lors de la campagne précédente 
        st.write("Répartition du deposit des personnes n'ayant pas été contactées lors de la campagne précédente :")
        df_previous = df[df["previous"]==0]
        st.write(df_previous.deposit.value_counts())
        
        #présentation en boxplot 
        fig_previous_2, ax = plt.subplots()
        sns.countplot(x="previous", hue="deposit", data=df_previous);
        ax.set_title("Répartition des clients n'ayant pas eu de contact lors de la dernière campagne en fonction du deposit", fontsize=9);
        st.write("Visualisation par graphique :")
        st.pyplot(fig_previous_2)
        
        # Focus sur les personnes ayant été contactées
        st.write("Focus sur les personnes ayant été contactées :")
        df_previous_max = df[df["previous"]>0]
        st.write(df_previous_max.deposit.value_counts())
        
        #présentation de graphie
        fig_previous_3 = sns.catplot(x="deposit", y="previous", kind="violin", data=df_previous_max);
        plt.title("Zoom sur les clients ayant été contactés au moins une fois");
        st.write("Visualisation par graphique :")
        st.pyplot(fig_previous_3)      
 
        st.write("Visiblement, les clients n’ayant pas eu de contact lors de la précédente campagne sont moins enclins à accepter la proposition de cette nouvelle campagne. Nous avons exactement la même répartition que sur la variable pdays. Cela conforte notre hypothèse qu’il s’agit de nouveaux clients.")
     
#deposit 
    if option == "deposit" :
        st.write("Présentation de la variable : _indique si le client a effectué un dépôt à terme lors de cette campagne Yes / No (variable catégorielle)_")
        st.write("C'est la variable cible de ce jeu de donnée.")
        
        #corrélation entre les variables cat, graphe heatmap
        st.write("Étude sur la corrélation des variables catégorielles par rapport à la variable cible :")
        cat_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan','month', 'contact', 'poutcome']
        corr_matrix = df[cat_vars + ['deposit']].apply(lambda x : pd.factorize(x)[0]).corr()
        fig_deposit, ax = plt.subplots()
        sns.heatmap(corr_matrix, cmap="Blues", annot=True, fmt=".2f", vmin=0, vmax=1)
        plt.figure(figsize = (14,8))
        ax.set_title("Corrélation entre les variables catégorielles et la variable cible", fontsize=16)  
        ax.set_xlabel("Variables catégorielles", fontsize=14)   
        ax.set_ylabel("Variables catégorielles", fontsize=14)
        st.pyplot(fig_deposit)
        st.write("La Heatmap indique que les corrélations entre les variables sont faibles. Essayons de voir en faisant une matrice de contingence.")

        #heatmap sur les variables numériques 
        st.write("Heatmap sur les variables numériques :")
        df_corr = df.corr()  
        fig_deposit_1, ax = plt.subplots(figsize = (7,7))  
        sns.heatmap(df_corr, annot=True, ax=ax, cmap="coolwarm");
        st.pyplot(fig_deposit_1)


# In[5]:


if page == pages[3]:
    st.title("Nos hypothèses d'études")
    st.header("Les variables 'day' et 'month' :")
    st.write("Ces variables auraient pu être très intéressantes pour étudier la saisonnalité de l'activité, les périodes où les clients seraient plus enclins à accepter une proposition d'un produit financier. Malheureusement sans l'année pour compléter ces 2 variables, nous manquons de précision.")
    st.write("En effet, la grande disparité entre les mois suscitent un certain nombre de question. Au final ces variables apportent plus de questions qu'elles n'apportent d'informations. De ce fait, nous avons décidé de ne pas les exploiter.")
    
    st.header("La variable 'default' :")
    st.write("Cette variable présente un déséquilibre trop important, en effet la modalité 'yes' possède une fréquence de 98%. Ainsi il est plus judicieux de ne pas l'inclure pour le machine learning car cela risque de biaiser les performances du modèle.")
    
    st.header("Traitement des modalités 'unknown' :")
    st.write("Concernant les variables 'job' et 'education', le nombre de cette modalité est respectivement de 70 et 497, ce qui représente 1% et 4% du dataset. Au vu de cette faible représentation, nous avons décidé de les remplacer par leur mode respectif.")
    st.write("Concernant la variable 'contact', cette variable se concentrant sur 3 modalités avec entre autre 'cellular' et 'telephone', nous avons retenu l'hypothèse que les 'unknown' indiquaient que l'organisme ne possédait pas le numéro de téléphone du client. Ainsi cela représente une information en soi. Nous avons donc décidé de conserver cette modalité en l'état.")
    st.write("Concernant la variable poutcome, nous pensons qu'elle est liée aux variables 'pdays' et 'previous'.")
    df_new_client = df[(df["previous"]==0) & (df["pdays"]==-1) & (df["poutcome"] == "unknown")]
    st.write("En effet, si nous filtrons le dataframe sur les modalités -1 de 'pdays', 0 de 'previous' et unknown de 'poutcome', nous avons", len(df_new_client),"lignes ce qui correspond pratiquement à la totalité des cas où 'poutcome' est 'unknown'. Nous émettons donc l'hypothèse que les clients ayant ces trois modalités sont tout simplement de nouveaux clients qui n'ont pas été contactés lors de la campagne précédente.")
    st.write("Ainsi ces 'unknown' représente une information, nous les conservons en l'état.")
    
    st.header("La variable duration")
    st.write("Cette variable possède la plus forte corrélation avec la variable cible, cependant, elle présente la particularité d'être connue seulement une fois l'action réalisée. Nous verrons par la suite que son influence est largement supérieure aux autres variables. Ainsi, nous avons décidé de créer deux ensembles de données d'entraînement et de test, l'un avec la variable 'duration' et l'autre sans. Nous laisserons le choix de l'inclure ou de l'exclure selon les préférences.")
    
    st.header("Traitement des modalités 'yes' / 'no' :")
    st.write("Ces modalités seront tout simplement remplacées par respectivement 1 et 0.")
    
    st.header("OneHotEncoding")
    st.write("Les variables 'job', 'marital', 'education', 'poutcome' et 'contact' seront encodés via ce procédé.")
    
    st.header("Normalisation")
    st.write("Les variables 'age' et 'duration' (pour le premier jeu) seront traitées via ce procédé.")
    
    st.header("Standardisation")
    st.write("Les variables 'balance', 'campaign', 'pdays' et 'previous' seront quant à elles standardisées.")


# In[6]:


# Preprocessing avec duration

# Séparation de la variable cible du reste du dataset
feats = df.drop("deposit", axis = 1)
target = df["deposit"]

# Suppression des variables month, day et default
feats = feats.drop(["month", "day", "default"], axis = 1)

# Séparation du jeu de données en un set d'entrainement et un jeu de test
X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.25, random_state = 42)

# Remplacement de la modalité unknown dans les variables Job et Education par leur mode respectif dans le jeu test et train
X_train["job"] = X_train["job"].replace("unknown", X_train["job"].mode()[0])
X_train["education"] = X_train["education"].replace("unknown", X_train["education"].mode()[0])
X_test["job"] = X_test["job"].replace("unknown", X_test["job"].mode()[0])
X_test["education"] = X_test["education"].replace("unknown", X_test["education"].mode()[0])

# Remplacement des yes et no sur les variables : housing, loan, deposit
X_train.replace(["yes", "no"], [1, 0], inplace=True)
X_test.replace(["yes", "no"], [1, 0], inplace=True)
y_train.replace(["yes", "no"], [1, 0], inplace=True)
y_test.replace(["yes", "no"], [1, 0], inplace=True)

# Transformation des variables catégorielles avec OneHotEncoding : job, marital, education, poutcome, contact
X_train_cat = X_train[["job", "marital", "education", "poutcome", "contact"]]
X_train = X_train.drop(["job", "marital", "education", "poutcome", "contact"], axis=1)
X_test_cat = X_test[["job", "marital", "education", "poutcome", "contact"]]
X_test = X_test.drop(["job", "marital", "education", "poutcome", "contact"], axis=1)
ohe = OneHotEncoder(drop=None, sparse=False)
X_train_ohe = pd.DataFrame(ohe.fit_transform(X_train_cat), columns = ohe.get_feature_names_out())
X_test_ohe = pd.DataFrame(ohe.transform(X_test_cat), columns = ohe.get_feature_names_out())

# Normalisation des variables : age, duration
norm = MinMaxScaler()
X_train[["age", "duration"]] = norm.fit_transform(X_train[["age", "duration"]])
X_test[["age", "duration"]] = norm.transform(X_test[["age", "duration"]])

# Standardisation des variables : balance, campaign, pdays, previous
scaler = StandardScaler()
X_train[["balance", "campaign", "pdays", "previous"]] = scaler.fit_transform(X_train[["balance", "campaign", "pdays", "previous"]])
X_test[["balance", "campaign", "pdays", "previous"]] = scaler.transform(X_test[["balance", "campaign", "pdays", "previous"]])

# Regroupement sous un seul DataFrame
# Reset des index
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
X_train_ohe = X_train_ohe.reset_index(drop=True)
X_test_ohe = X_test_ohe.reset_index(drop=True)

# Concaténation
X_train_prepared = pd.concat([X_train, X_train_ohe], axis=1)
X_test_prepared = pd.concat([X_test, X_test_ohe], axis=1)


# In[8]:


# Preprocessing sans duration

# Séparation de la variable cible du reste du dataset
feats_2 = df.drop("deposit", axis = 1)
target_2 = df["deposit"]

# Suppression des variables month, day et default
feats_2 = feats_2.drop(["month", "day", "default", "duration"], axis = 1)

# Séparation du jeu de données en un set d'entrainement et un jeu de test
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(feats_2, target_2, test_size = 0.25, random_state = 42)

# Remplacement de la modalité unknown dans les variables Job et Education par leur mode respectif dans le jeu test et train
X_train_2["job"] = X_train_2["job"].replace("unknown", X_train_2["job"].mode()[0])
X_train_2["education"] = X_train_2["education"].replace("unknown", X_train_2["education"].mode()[0])
X_test_2["job"] = X_test_2["job"].replace("unknown", X_test_2["job"].mode()[0])
X_test_2["education"] = X_test_2["education"].replace("unknown", X_test_2["education"].mode()[0])

# Remplacement des yes et no sur les variables : housing, loan, deposit
X_train_2.replace(["yes", "no"], [1, 0], inplace=True)
X_test_2.replace(["yes", "no"], [1, 0], inplace=True)
y_train_2.replace(["yes", "no"], [1, 0], inplace=True)
y_test_2.replace(["yes", "no"], [1, 0], inplace=True)

# Transformation des variables catégorielles avec OneHotEncoding : job, marital, education, poutcome, contact
X_train_2_cat = X_train_2[["job", "marital", "education", "poutcome", "contact"]]
X_train_2 = X_train_2.drop(["job", "marital", "education", "poutcome", "contact"], axis=1)
X_test_2_cat = X_test_2[["job", "marital", "education", "poutcome", "contact"]]
X_test_2 = X_test_2.drop(["job", "marital", "education", "poutcome", "contact"], axis=1)
ohe_2 = OneHotEncoder(drop=None, sparse=False)
X_train_ohe_2 = pd.DataFrame(ohe_2.fit_transform(X_train_cat), columns = ohe_2.get_feature_names_out())
X_test_ohe_2 = pd.DataFrame(ohe_2.transform(X_test_cat), columns = ohe_2.get_feature_names_out())

# Normalisation des variables : age
norm_2 = MinMaxScaler()
X_train_2["age"] = norm_2.fit_transform(X_train_2[["age"]])
X_test_2["age"] = norm_2.transform(X_test_2[["age"]])

# Standardisation des variables : balance, campaign, pdays, previous
scaler_2 = StandardScaler()
X_train_2[["balance", "campaign", "pdays", "previous"]] = scaler_2.fit_transform(X_train_2[["balance", "campaign", "pdays", "previous"]])
X_test_2[["balance", "campaign", "pdays", "previous"]] = scaler_2.transform(X_test_2[["balance", "campaign", "pdays", "previous"]])

# Regroupement sous un seul DataFrame
# Reset des index
X_train_2 = X_train_2.reset_index(drop=True)
X_test_2 = X_test_2.reset_index(drop=True)
X_train_ohe_2 = X_train_ohe_2.reset_index(drop=True)
X_test_ohe_2 = X_test_ohe_2.reset_index(drop=True)

# Concaténation
X_train_prepared_2 = pd.concat([X_train_2, X_train_ohe_2], axis=1)
X_test_prepared_2 = pd.concat([X_test_2, X_test_ohe_2], axis=1)


# In[10]:


if page == pages[4]:
    st.title('Sélection du modèle de machine learning')

    selected_box = st.selectbox('Choix du modèle', [' ', 'Random Forest', 'Gradient Boosting Classifier'])
    st.write("   ")  # Adds a blank line
    st.write("   ")  # Adds a blank line
    st.write("   ")  # Adds a blank line

    if selected_box == 'Random Forest':
        selected_duration = st.selectbox("Sélection du dataframe", ["Avec 'duration'", "Sans 'duration'"])
        if selected_duration == "Avec 'duration'" :
            parameters = st.select_slider('Select parameters', options=['Standard', 'Optimisés'])
            if parameters == "Standard" :            
                if st.button('Lancer la simulation'):
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X_train_prepared, y_train)
                    preds_rf = rf.predict(X_test_prepared)
                    st.write("Matrice de confusion :")
                    st.write(pd.crosstab(y_test, preds_rf, rownames=["Réalité"], colnames=["Prédictions"]))
                    accuracy = accuracy_score(y_test, preds_rf)
                    recall = recall_score(y_test, preds_rf)
                    f1 = f1_score(y_test, preds_rf)
                    score_rf = pd.DataFrame({"Accuracy" : [accuracy],
                                                "Recall" : [recall],
                                                "f1" : [f1]})
                    st.write("Tableau récapitulatif des scores du modèle :")
                    st.write(score_rf)
                    importances = rf.feature_importances_
                    importance_df = pd.DataFrame({'feature': X_train_prepared.columns, 'importance': importances})
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    fig_rf = importance_df.plot(kind='bar', x='feature', y='importance')
                    plt.title('Importance des variables')
                    plt.xlabel('Variables')
                    plt.ylabel('Importance')
                    st.write("Classement par ordre décroissant de l'importance des variables dans ce modèle :")
                    st.pyplot(fig_rf.get_figure())

                    importances_tree = []
                    for tree in rf.estimators_:
                        importances_tree.append(tree.feature_importances_)
                    importances_array = np.array(importances_tree)
                    if importances_array.ndim == 1:
                        importances_array = importances_array.T
                    importance_tree = pd.DataFrame(importances_array, columns=X_train_prepared.columns)
                    st.write("Importance des variables pour chaque arbre du random en fonction de la moyenne de Gini :")
                    st.write(importance_tree)
                        
            if parameters == "Optimisés" :            
                if st.button('Lancer la simulation'):
                    st.write("Voici les meilleurs paramètres trouvés via GridSearchCV :")
                    st.write("'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200")
                    rf = RandomForestClassifier(max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200, random_state=42)
                    rf.fit(X_train_prepared, y_train)
                    preds_rf = rf.predict(X_test_prepared)
                    st.write("Matrice de confusion :")
                    st.write(pd.crosstab(y_test, preds_rf, rownames=["Réalité"], colnames=["Prédictions"]))
                    accuracy = accuracy_score(y_test, preds_rf)
                    recall = recall_score(y_test, preds_rf)
                    f1 = f1_score(y_test, preds_rf)
                    score_rf = pd.DataFrame({"Accuracy" : [accuracy],
                                                "Recall" : [recall],
                                                "f1" : [f1]})
                    st.write("Tableau récapitulatif des scores du modèle :")
                    st.write(score_rf)
                    importances = rf.feature_importances_
                    importance_df = pd.DataFrame({'feature': X_train_prepared.columns, 'importance': importances})
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    fig_rf = importance_df.plot(kind='bar', x='feature', y='importance')
                    plt.title('Importance des variables')
                    plt.xlabel('Variables')
                    plt.ylabel('Importance')
                    st.write("Classement par ordre décroissant de l'importance des variables dans ce modèle :")
                    st.pyplot(fig_rf.get_figure())

                    importances_tree = []
                    for tree in rf.estimators_:
                        importances_tree.append(tree.feature_importances_)
                    importances_array = np.array(importances_tree)
                    if importances_array.ndim == 1:
                        importances_array = importances_array.T
                    importance_tree = pd.DataFrame(importances_array, columns=X_train_prepared.columns)
                    st.write("Importance des variables pour chaque arbre du random en fonction de la moyenne de Gini :")
                    st.write(importance_tree)    
                
        if selected_duration == "Sans 'duration'" :
            parameters = st.select_slider('Select parameters', options=['Standard', 'Optimisés'])
            if parameters == "Standard" :            
                if st.button('Lancer la simulation'):
                    rf_2 = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf_2.fit(X_train_prepared_2, y_train_2)
                    preds_rf_2 = rf_2.predict(X_test_prepared_2)
                    st.write("Matrice de confusion :")
                    st.write(pd.crosstab(y_test_2, preds_rf_2, rownames=["Réalité"], colnames=["Prédictions"]))
                    accuracy_2 = accuracy_score(y_test_2, preds_rf_2)
                    recall_2 = recall_score(y_test_2, preds_rf_2)
                    f1_2 = f1_score(y_test_2, preds_rf_2)
                    score_rf_2 = pd.DataFrame({"Accuracy" : [accuracy_2],
                                            "Recall" : [recall_2],
                                            "f1" : [f1_2]})
                    st.write("Tableau récapitulatif des scores du modèle :")
                    st.write(score_rf_2)
                    importances = rf_2.feature_importances_
                    importance_df = pd.DataFrame({'feature': X_train_prepared_2.columns, 'importance': importances})
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    fig_rf = importance_df.plot(kind='bar', x='feature', y='importance')
                    plt.title('Importance des variables')
                    plt.xlabel('Variables')
                    plt.ylabel('Importance')
                    st.write("Classement par ordre décroissant de l'importance des variables dans ce modèle :")
                    st.pyplot(fig_rf.get_figure())

                    importances_tree = []
                    for tree in rf_2.estimators_:
                        importances_tree.append(tree.feature_importances_)
                    importances_array = np.array(importances_tree)
                    if importances_array.ndim == 1:
                        importances_array = importances_array.T
                    importance_tree = pd.DataFrame(importances_array, columns=X_train_prepared_2.columns)
                    st.write("Importance des variables pour chaque arbre du random en fonction de la moyenne de Gini :")
                    st.write(importance_tree)
                    
            if parameters == "Optimisés" :            
                if st.button('Lancer la simulation'):
                    st.write("Voici les meilleurs paramètres trouvés via GridSearchCV :")
                    st.write("'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200")
                    rf_2 = RandomForestClassifier(max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=200, random_state=42)
                    rf_2.fit(X_train_prepared_2, y_train_2)
                    preds_rf_2 = rf_2.predict(X_test_prepared_2)
                    st.write("Matrice de confusion :")
                    st.write(pd.crosstab(y_test_2, preds_rf_2, rownames=["Réalité"], colnames=["Prédictions"]))
                    accuracy_2 = accuracy_score(y_test_2, preds_rf_2)
                    recall_2 = recall_score(y_test_2, preds_rf_2)
                    f1_2 = f1_score(y_test_2, preds_rf_2)
                    score_rf_2 = pd.DataFrame({"Accuracy" : [accuracy_2],
                                            "Recall" : [recall_2],
                                            "f1" : [f1_2]})
                    st.write("Tableau récapitulatif des scores du modèle :")
                    st.write(score_rf_2)
                    importances = rf_2.feature_importances_
                    importance_df = pd.DataFrame({'feature': X_train_prepared_2.columns, 'importance': importances})
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    fig_rf = importance_df.plot(kind='bar', x='feature', y='importance')
                    plt.title('Importance des variables')
                    plt.xlabel('Variables')
                    plt.ylabel('Importance')
                    st.write("Classement par ordre décroissant de l'importance des variables dans ce modèle :")
                    st.pyplot(fig_rf.get_figure())

                    importances_tree = []
                    for tree in rf_2.estimators_:
                        importances_tree.append(tree.feature_importances_)
                    importances_array = np.array(importances_tree)
                    if importances_array.ndim == 1:
                        importances_array = importances_array.T
                    importance_tree = pd.DataFrame(importances_array, columns=X_train_prepared_2.columns)
                    st.write("Importance des variables pour chaque arbre du random en fonction de la moyenne de Gini :")
                    st.write(importance_tree)

    if selected_box == 'Gradient Boosting Classifier':
        selected_duration = st.selectbox("Sélection du dataframe", ["Avec 'duration'", "Sans 'duration'"])
        if selected_duration == "Avec 'duration'" :
            parameters = st.select_slider('Select parameters', options=['Standard', 'Optimisés'])
            if parameters == "Standard" :    
                if st.button('Lancer la simulation'):
                    gbc = GradientBoostingClassifier(random_state=42)
                    gbc.fit(X_train_prepared, y_train)
                    score = gbc.score(X_test_prepared, y_test)
                    gbc_pred = gbc.predict(X_test_prepared)
                    st.write("Matrice de confusion :")
                    st.write(pd.crosstab(y_test, gbc_pred, rownames=["Réalité"], colnames=["Prédictions"]))
                    gbc_f1 = f1_score(y_test, gbc_pred)
                    gbc_recall = recall_score(y_test, gbc_pred)
                    gbc_score = pd.DataFrame({"Accuracy" : [gbc_f1],
                                            "Recall" : [gbc_recall],
                                            "f1" : [gbc_f1]})
                    st.write("Tableau récapitulatif des scores du modèle :")
                    st.write(gbc_score)
                    importances_gbc = gbc.feature_importances_
                    importance_df_gbc = pd.DataFrame({'feature': X_train_prepared.columns, 'importance': importances_gbc})
                    importance_df = importance_df_gbc.sort_values('importance', ascending=False)
                    fig_gbc = importance_df.plot(kind='bar', x='feature', y='importance')
                    plt.title('Importance des variables')
                    plt.xlabel('Variables')
                    plt.ylabel('Importance')
                    st.write("Classement par ordre décroissant de l'importance des variables dans ce modèle :")
                    st.pyplot(fig_gbc.get_figure())

                    # Prédiction des probabilités de la classe positive
                    y_pred_prob = gbc.predict_proba(X_test_prepared)[:, 1]

                    # Calcul de la courbe ROC
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)

                    # Tracé de la courbe ROC
                    import matplotlib.pyplot as plt
                    plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Taux de faux positifs')
                    plt.ylabel('Taux de vrais positifs')
                    plt.title('Courbe ROC')
                    plt.legend(loc="lower right")

                    # Affichage de la courbe ROC dans Streamlit
                    st.write("Graphique ROC  (Receiver Operating Characteristic):")
                    st.pyplot(plt)
                    
            if parameters == "Optimisés" :    
                if st.button('Lancer la simulation'):
                    st.write("Voici les meilleurs paramètres trouvés via GridSearchCV :")
                    st.write("'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200, 'subsample': 0.8")
                    gbc = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, min_samples_leaf=1, min_samples_split=2, n_estimators=200, subsample=0.8, random_state=42)
                    gbc.fit(X_train_prepared, y_train)
                    score = gbc.score(X_test_prepared, y_test)
                    gbc_pred = gbc.predict(X_test_prepared)
                    st.write("Matrice de confusion :")
                    st.write(pd.crosstab(y_test, gbc_pred, rownames=["Réalité"], colnames=["Prédictions"]))
                    gbc_f1 = f1_score(y_test, gbc_pred)
                    gbc_recall = recall_score(y_test, gbc_pred)
                    gbc_score = pd.DataFrame({"Accuracy" : [gbc_f1],
                                            "Recall" : [gbc_recall],
                                            "f1" : [gbc_f1]})
                    st.write("Tableau récapitulatif des scores du modèle :")
                    st.write(gbc_score)
                    importances_gbc = gbc.feature_importances_
                    importance_df_gbc = pd.DataFrame({'feature': X_train_prepared.columns, 'importance': importances_gbc})
                    importance_df = importance_df_gbc.sort_values('importance', ascending=False)
                    fig_gbc = importance_df.plot(kind='bar', x='feature', y='importance')
                    plt.title('Importance des variables')
                    plt.xlabel('Variables')
                    plt.ylabel('Importance')
                    st.write("Classement par ordre décroissant de l'importance des variables dans ce modèle :")
                    st.pyplot(fig_gbc.get_figure())

                    # Prédiction des probabilités de la classe positive
                    y_pred_prob = gbc.predict_proba(X_test_prepared)[:, 1]

                    # Calcul de la courbe ROC
                    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
                    roc_auc = auc(fpr, tpr)

                    # Tracé de la courbe ROC
                    import matplotlib.pyplot as plt
                    plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Taux de faux positifs')
                    plt.ylabel('Taux de vrais positifs')
                    plt.title('Courbe ROC')
                    plt.legend(loc="lower right")

                    # Affichage de la courbe ROC dans Streamlit
                    st.write("Graphique ROC  (Receiver Operating Characteristic):")
                    st.pyplot(plt)
                
                
        if selected_duration == "Sans 'duration'" :
            parameters = st.select_slider('Select parameters', options=['Standard', 'Optimisés'])
            if parameters == "Standard" : 
                if st.button('Lancer la simulation'):
                    gbc = GradientBoostingClassifier(random_state=42)
                    gbc.fit(X_train_prepared_2, y_train_2)
                    score = gbc.score(X_test_prepared_2, y_test_2)
                    gbc_pred = gbc.predict(X_test_prepared_2)
                    st.write("Matrice de confusion :")
                    st.write(pd.crosstab(y_test_2, gbc_pred, rownames=["Réalité"], colnames=["Prédictions"]))
                    gbc_f1 = f1_score(y_test_2, gbc_pred)
                    gbc_recall = recall_score(y_test_2, gbc_pred)
                    gbc_score = pd.DataFrame({"Accuracy" : [gbc_f1],
                                            "Recall" : [gbc_recall],
                                            "f1" : [gbc_f1]})
                    st.write("Tableau récapitulatif des scores du modèle :")
                    st.write(gbc_score)
                    importances_gbc = gbc.feature_importances_
                    importance_df_gbc = pd.DataFrame({'feature': X_train_prepared_2.columns, 'importance': importances_gbc})
                    importance_df = importance_df_gbc.sort_values('importance', ascending=False)
                    fig_gbc = importance_df.plot(kind='bar', x='feature', y='importance')
                    plt.title('Importance des variables')
                    plt.xlabel('Variables')
                    plt.ylabel('Importance')
                    st.write("Classement par ordre décroissant de l'importance des variables dans ce modèle :")
                    st.pyplot(fig_gbc.get_figure())

                    # Prédiction des probabilités de la classe positive
                    y_pred_prob = gbc.predict_proba(X_test_prepared_2)[:, 1]

                    # Calcul de la courbe ROC
                    fpr, tpr, thresholds = roc_curve(y_test_2, y_pred_prob)
                    roc_auc = auc(fpr, tpr)

                    # Tracé de la courbe ROC
                    import matplotlib.pyplot as plt
                    plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Taux de faux positifs')
                    plt.ylabel('Taux de vrais positifs')
                    plt.title('Courbe ROC')
                    plt.legend(loc="lower right")

                    # Affichage de la courbe ROC dans Streamlit
                    st.write("Graphique ROC  (Receiver Operating Characteristic):")
                    st.pyplot(plt)
            
            if parameters == "Optimisés" : 
                if st.button('Lancer la simulation'):
                    st.write("Voici les meilleurs paramètres trouvés via GridSearchCV :")
                    st.write("'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200, 'subsample': 0.8")
                    gbc = GradientBoostingClassifier(learning_rate=0.1, max_depth=3, min_samples_leaf=1, min_samples_split=2, n_estimators=200, subsample=0.8, random_state=42)
                    gbc.fit(X_train_prepared_2, y_train_2)
                    score = gbc.score(X_test_prepared_2, y_test_2)
                    gbc_pred = gbc.predict(X_test_prepared_2)
                    st.write("Matrice de confusion :")
                    st.write(pd.crosstab(y_test_2, gbc_pred, rownames=["Réalité"], colnames=["Prédictions"]))
                    gbc_f1 = f1_score(y_test_2, gbc_pred)
                    gbc_recall = recall_score(y_test_2, gbc_pred)
                    gbc_score = pd.DataFrame({"Accuracy" : [gbc_f1],
                                            "Recall" : [gbc_recall],
                                            "f1" : [gbc_f1]})
                    st.write("Tableau récapitulatif des scores du modèle :")
                    st.write(gbc_score)
                    importances_gbc = gbc.feature_importances_
                    importance_df_gbc = pd.DataFrame({'feature': X_train_prepared_2.columns, 'importance': importances_gbc})
                    importance_df = importance_df_gbc.sort_values('importance', ascending=False)
                    fig_gbc = importance_df.plot(kind='bar', x='feature', y='importance')
                    plt.title('Importance des variables')
                    plt.xlabel('Variables')
                    plt.ylabel('Importance')
                    st.write("Classement par ordre décroissant de l'importance des variables dans ce modèle :")
                    st.pyplot(fig_gbc.get_figure())

                    # Prédiction des probabilités de la classe positive
                    y_pred_prob = gbc.predict_proba(X_test_prepared_2)[:, 1]

                    # Calcul de la courbe ROC
                    fpr, tpr, thresholds = roc_curve(y_test_2, y_pred_prob)
                    roc_auc = auc(fpr, tpr)

                    # Tracé de la courbe ROC
                    import matplotlib.pyplot as plt
                    plt.figure()
                    lw = 2
                    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='Courbe ROC (AUC = %0.2f)' % roc_auc)
                    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Taux de faux positifs')
                    plt.ylabel('Taux de vrais positifs')
                    plt.title('Courbe ROC')
                    plt.legend(loc="lower right")

                    # Affichage de la courbe ROC dans Streamlit
                    st.write("Graphique ROC  (Receiver Operating Characteristic):")
                    st.pyplot(plt)


# In[11]:


# Préparation de la page simulation
job_value = df.job.unique().tolist()
job_value.pop(9)
marital_value = df.marital.unique().tolist()
education_value = df.education.unique().tolist()
education_value.pop(3)
housing_value = df.housing.unique().tolist()
loan_value = df.loan.unique().tolist()
contact_value = df.contact.unique().tolist()
poutcome_value = df.poutcome.unique().tolist()


# In[12]:


if page == pages[5]:
    st.title("Simulation avec Gradient Boosting Classifier sans 'duration'")
    st.header("Veuillez renseigner les informations sur le client :")
    # Questionnaire
    value_age = st.number_input("L'âge :", min_value=18, max_value=100, value=25, step=1)
    value_job = st.selectbox('Situation professionnelle :', job_value)
    value_marital = st.selectbox('Situation maritale :', marital_value)
    value_education = st.selectbox("Niveau d'études :", education_value)
    value_balance = st.number_input("Solde du compte :", min_value=-100000, max_value=1000000, value=0, step=1)
    value_housing = st.selectbox("Propriétaire du logement principal :", housing_value)
    value_loan = st.selectbox("Prêt personnel en cours :", loan_value)
    value_contact = st.selectbox("Moyen de communication avec le client :", contact_value)
    value_campaign = st.number_input("Nombre de contact lors de cette campagne :", min_value=0, max_value=100, value=0, step=1)
    value_pdays = st.number_input("Nombre de jours écoulés depuis le dernier contact :", min_value=-1, max_value=1000, value=-1, step=1)
    value_previous = st.number_input("Nombre de contact établis avec le client lors de la précédente campagne :", min_value=0, max_value=100, value=0, step=1)
    value_poutcome = st.selectbox("Résultat de la précédente campagne :", poutcome_value)
    

    if st.button('Lancer la simulation'):
            # Création du dataframe du formulaire
        df_formulaire = pd.DataFrame({"age" : [value_age],
                                     "job" : [value_job],
                                     "marital" : [value_marital],
                                     "education" : [value_education],
                                     "balance" : [value_balance],
                                     "housing" : [value_housing],
                                     "loan" : [value_loan],
                                     "contact" : [value_contact],
                                     "campaign" : [value_campaign],
                                     "pdays" : [value_pdays],
                                     "previous" : [value_previous],
                                     "poutcome" : [value_poutcome]})

        # Preprocessing formulaire

        # Remplacement des yes et no sur les variables : housing, loan
        df_formulaire.replace(["yes", "no"], [1, 0], inplace=True)

        # Transformation des variables catégorielles avec OneHotEncoding : job, marital, education, poutcome, contact
        df_formulaire_cat = df_formulaire[["job", "marital", "education", "poutcome", "contact"]]
        df_formulaire = df_formulaire.drop(["job", "marital", "education", "poutcome", "contact"], axis=1)
        df_formulaire_ohe = pd.DataFrame(ohe_2.transform(df_formulaire_cat), columns = ohe_2.get_feature_names_out())

        # Normalisation des variables : age, duration
        df_formulaire["age"] = norm_2.transform(df_formulaire[["age"]])

        # Standardisation des variables : balance, campaign, pdays, previous
        df_formulaire[["balance", "campaign", "pdays", "previous"]] = scaler_2.transform(df_formulaire[["balance", "campaign", "pdays", "previous"]])

        # Regroupement sous un seul DataFrame
        # Reset des index
        df_formulaire = df_formulaire.reset_index(drop=True)
        df_formulaire_ohe = df_formulaire_ohe.reset_index(drop=True)

        # Concaténation
        df_formulaire_prepared = pd.concat([df_formulaire, df_formulaire_ohe], axis=1)

        model_random =  GradientBoostingClassifier( n_estimators=200, subsample=0.5, max_depth=10, random_state=42)
        # Entrainement
        model_random.fit(X_train_prepared_2, y_train_2)

        # Prédictions 
        preds_random = model_random.predict(X_test_prepared_2) 

        # Prédiction du modèle
        formulaire_random = model_random.predict(df_formulaire_prepared)
        
        if formulaire_random[0] == 1:
            st.success("Votre client a de fortes chances de souscrire à votre offre.")
        else :
            st.warning("Il est peu probable que le client souscrive à votre offre.")

