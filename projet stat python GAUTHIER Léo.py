###################################################################################################################
#
# Préparation : importation modules
#
####################################################################################################################

# Modules importés

import os
import pandas as pds 
import numpy as np 
from scipy import stats
import statsmodels.api as sm #module qqplot
import matplotlib.pyplot as plt #module qqplot 
from statsmodels.formula.api import ols #module mco
import statsmodels.stats.multicomp as mc #module multicomparaison anova

# Changement répertoire

os.getcwd()
chemin = r"C:\Users\leoga\Desktop\M1\Statistique avec python\Projet python"
os.chdir(chemin)
os.getcwd()

#On suppose travailler au seuil 5% tout au long du projet, sauf si mention contraire

###################################################################################################################
#
# Exercice 1 : test d'indépendance
#
####################################################################################################################


# Création du jeu de données

obs = np.array([[592, 119, 849, 504, 36], [544, 97, 677, 451, 14]])
chi2, p, dof, thq = stats.chi2_contingency(obs, correction=False)
print(thq)

#Avec le print(thq) on vérifie que la règle de Cochran soit bien vérifiée, ce qui est le cas puisque aucune valeur n'est inférieur à 5.

# On peut également la vérifiée avec une boucle :

thq.shape[1]

Coch=0
for i in range(thq.shape[0]):
    for j in range(thq.shape[1]):
        if thq[i,j]<5:
            Coch+=1
percent=Coch/(thq.shape[0]*thq.shape[1])*100
print('Il y a', '{:.2f}'.format(percent), '% des effectifs théoriques strictement inférieur à 5')

#Puisque la règle de cochran est vérifiée, on peut passer à l'intérpétation des résultats du test du khié2 d'indépendance:

print("La statistique de test vaut", '{:.2f}'.format(chi2),"et sous H0, elle suit une loi du khi2 à", dof,"degrés de liberté. La p-valeur est", '{:.4f}'.format(p))

#Le test du Khi-2 de Pearson utilise une approximation de la log-vraisemblance, ce qui rend le résultat moins précis.
#Il existe le G-Test qui utilise la valeur exacte de la log-vraisemblance.
#Il s'applique sous les mêmes conditions d'application que le test du khi-2.
#Sous H0, la stat du g-test suit une khi2 avec le même nombre de ddl que dans le test du khi2

g, pg, dof_g, thq_g = stats.chi2_contingency(obs, lambda_="log-likelihood", correction=False)

print("La statistique de test vaut", '{:.2f}'.format(g),"et sous H0, elle suit une loi du khi2 à", dof_g,"degrés de liberté. La p-valeur est", '{:.4f}'.format(pg))



###################################################################################################################
#
# Exercice 2 : Test de Student
#
####################################################################################################################


man=pds.read_csv("Man.csv", sep=";", decimal=",")

#étape 1 : les statistiques descriptives

man.info()
man.describe()
#Le fichier contient 27 observations, mais les manalas ont 3 données manquantes
#Il faudra y faire attention 

#étape 2 : vérification des conditions d'application du test de Student
#On veut tester si le poid moyen des maneles n'est pas significativement différent du poids des manalas, à un aléa d'échantillonage près
#Pour cela on fait un test de Student
#Il y 5 conditions d'application à vérifier pour pouvoir appliquer le test de Student:

#1) les observations au sein de l'échantillon sont indépendantes : j'espère
#2) les échantillons sont mutuellement indépendants : j'espère
#3) les échantillons 1 et 2 suivent chacuns une loi gaussienne : test de shapiro wilk
#4) les écarts types sont inconnus : c'est le cas car on ne connait pas le poids moyen donc connaître l'écart type risque d'être complexe
#5) les écarts types sont égaux : test d'égalité des variance de fisher

#On fait le test de shapiro wilk, dont la condition d'application est l'indépendance des observations
#On a supposé que c'était le cas, donc on peut réaliser le test

ssw1, psw1 = stats.shapiro(man["Manele"])
print("Sous H0, la statistique de test vaut", '{:.2f}'.format(ssw1),".La p-valeur vaut", '{:.4f}'.format(psw1), "et est supérieure à 0,05, donc H0 est accepté: l'échantillon concernant les maneles est distribué selon une loi normale")
#On peut aussi faire un Q-Q plot pour vérifier la normalité
sm.qqplot(man["Manele"], loc=np.mean(man["Manele"]), scale=np.std(man["Manele"]), line='45') 


ssw2, psw2 = stats.shapiro(man["Manala"].dropna())
sm.qqplot(man["Manala"].dropna(), loc=np.mean(man["Manala"].dropna()), scale=np.std(man["Manala"].dropna()), line='45')  
print("Sous H0, la statistique de test vaut", '{:.2f}'.format(ssw2),".La p-valeur vaut", '{:.4f}'.format(psw2), "et est supérieure à 0,05, donc H0 est accepté: l'échantillon concernant les manalas est distribué selon une loi normale")

#Nos deux échantillons sont donc bien distribués selon une loi normale

#On peut donc se lancer dans le test d'égalité des variances de fisher.
#Ce test possède également des conditions d'applications:
#1) les observations sont indé : OUI
#2) les échantillons sont indé : OUI
#3) l'échantillon 1 est distribué selon une normale : cf shapiro wilk
#4) l'échantillon 2 est distribué selon une loi normale : cf shapiro wilk

#Comme toutes les conditions d'application sont vérifiées, on peut se lancer dans le test de fisher

def f_test(x,y):
    x=np.array(x) #array -> mettre x dans un tableau numpy (pour calculer variance avec np ça marche pas)
    y=np.array(y)
    f=np.var(x, ddof=1)/np.var(y, ddof=1) #calcul stat de décision, ddof=1 pour corriger l'estimateur non biaisé
    dfn = x.size - 1 #ddl du numérateur
    dfd = y.size -1 #ddl du déno
    p = 1-stats.f.cdf(f, dfn, dfd) #calcul p value
    return f, p, dfn, dfd

f, pf, dfn, dfd = f_test(man["Manele"], man["Manala"].dropna())

print("La stat de test vaut F =",'{:.2f}'.format(f),".Sous H0, elle suit une loi de Fisher à", dfn,"et", dfd,"degrés de liberté. La p-valeur vaut", '{:.4f}'.format(pf),"et est supérieure à 0,05 donc H0 est acceptée: les écarts types des échantillons sont égaux")

#Puisque toutes les conditions d'application sont vérifiées, on fait le test de Student

st,pt=stats.ttest_ind(man["Manele"],man["Manala"],nan_policy='omit')
print("La statistique de test T vaut", '{:.2f}'.format(st), "et suit sous H0 une loi de Student à", len(man["Manele"])+len(man["Manala"].dropna())-2, " degrés de liberté. La p-valeur vaut", '{:.4f}'.format(pt))

###################################################################################################################
#
# Exercice 3 : ANOVA 
#
####################################################################################################################

tim=pds.read_csv("timbres.csv", sep=";", decimal=",")

tim.info()
tim.describe()
#Il n'y a aucune valeurs manquantes dans le jeu de données
#on crée le tableau d'effectifs du jeu par pays
tim_freq=pds.crosstab(tim.pays, "freq")
print(tim_freq)
#on crée ensuite les différents groupes de timbres par pays :
timAl = tim["epaisseur"][tim["pays"]=="Allemagne"]
timAu = tim["epaisseur"][tim["pays"]=="Autriche"]
timB = tim["epaisseur"][tim["pays"]=="Belgique"]
timF = tim["epaisseur"][tim["pays"]=="France"]

#On peut vérifier la longueur de ces 4 groupes pour voir si ça correspond
len(timAl)
len(timAu)
len(timB)
len(timF)
#Tout correspond

#On checke les statistiques descriptives 
timAl.describe()
timAu.describe()
timB.describe()
timF.describe()

#On crée le modèle pour pouvoir vérifier les conditions d'application direct sur les résidus
model = ols(' epaisseur ~ pays', data=tim).fit()

#On fait le test de normalité sur les résidus
model.resid
ssw_r, psw_r = stats.shapiro(model.resid)
print("Sous H0, la statistique de test vaut", '{:.2f}'.format(ssw_r),".La p-valeur vaut", '{:.4f}'.format(psw_r), "et est supérieure à 0,05, donc H0 est accepté: l'échantillon d'observation des résidus est distribué selon une loi normale")

#On vérifie désormais l'égalité des variances
#On fait pour ceci le test de Bartlett

B,pb=stats.bartlett(timAl, timAu, timB, timF)
print("La stat de décision B est",'{:.2f}'.format(B),".Sous H0 elle suit une khi deux à 3 ddl. La p-valeur est", '{:.4f}'.format(pb), ", supérieure à 0,05 donc H0 est acceptée: les 4 variances sont toutes égales")

aov_table = sm.stats.anova_lm(model, type=2)
print(aov_table)

print("L'épaisseur moyenne des timbres ne sont pas toutes égales (ANOVA, F = 194,62, p-value=5,82e-38")

#On va donc faire des comparaisons multiples avec le groupe de réference de timbres français
#On a 3 tests de Student à faire, les conditions d'application étant toute vérifiées précédemment
#Avant ça, on doit corriger les p-values
test = tim.filter(items =['epaisseur', 'pays'])
comp = mc.MultiComparison(test['epaisseur'], test['pays'])

tbl,a,b = comp.allpairtest(stats.ttest_ind, method = "bonf")

print(tbl)

#Puisqu'on fait des comparaisons multiples avec un groupe de référence, on ne s'intéresse pas à la colonne pval_corr (c dans le cas où on fait comparaison multiple sans grp de ref)
#on va plutôt multiplier la colonne pval par le nombre de test fait (ici 3)
#et on prend min(1, 3*pval)
#Ici évidemment les 3 p-val qui nous intéressent (celles impliquant la france) sont nulles (arrondies à 0)
#donc la p-val pour nos 3 tests de Student vaudra 3

#Test student moyenne allemagne / france

st_alf,pt_alf=stats.ttest_ind(timAl,timF)
print("La statistique de test T vaut", '{:.2f}'.format(st_alf), "et suit sous H0 une loi de Student à", len(timAl)+len(timF)-2, " degrés de liberté. La p-valeur vaut", '{:.4f}'.format(pt_alf),"< au seuil donc on rejette H0")

#Test student moyenne autriche / france

st_auf,pt_auf=stats.ttest_ind(timAu,timF)
print("La statistique de test T vaut", '{:.2f}'.format(st_auf), "et suit sous H0 une loi de Student à", len(timAu)+len(timF)-2, " degrés de liberté. La p-valeur vaut", '{:.4f}'.format(pt_auf), "< seuil donc on rejette H0")

#Test student moyenne belgique / france 

st_bf,pt_bf=stats.ttest_ind(timB,timF)
print("La statistique de test T vaut", '{:.2f}'.format(st_bf), "et suit sous H0 une loi de Student à", len(timB)+len(timF)-2, " degrés de liberté. La p-valeur vaut", '{:.4f}'.format(pt_bf),"< seuil donc on rejette H0")

#On fait pour conclure des comparaisons 2 à 2 sans groupe de références
#On utilise la correction de Tukey car moins conservatrice

post_hoc_res = comp.tukeyhsd()
print(post_hoc_res)

post_hoc_res.plot_simultaneous(ylabel = "pays", xlabel = "epaisseur")
































