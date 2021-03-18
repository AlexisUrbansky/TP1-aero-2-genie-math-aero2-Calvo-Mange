#################################### Propriété #############################################

    #Tp1 Génie mathématiques, méthode du pivot de Gauss et autre
    #Louis Calvo & Alexis Mangé

################################## Librairies importées ####################################

import numpy as np 
import time as t
from math import *
import matplotlib.pyplot as plt


###################### Définition du générateur de matrice aléatoire ####################### 

X = list()  
def random_mat(n):
    '''
    Cette fonction permet de générer deux matrices aléatoire A et B.
    Elle retourne ces deux matrices.
    '''

    A = np.random.rand(n, n)
    B = np.random.rand(n)
    if np.linalg.det(A) != 0:
        return A, B
    else:
        random_mat(n)
    
###################### Définition des fonctions pour pivot de Gauss ########################

def ResolutionSystTriSup(Taug):

    '''
    Cette fonction permet de résoudre un système à partir d'une matrice triangulaire supérieur de dimmension (n,n+1)
    Elle retourne une matrice contenant les solutions du système.
    '''

    global nb_ligne, nb_colonnes
    comptage=0
    liste_solutions = []
    v = Taug[nb_ligne-1][nb_colonnes-1]/Taug[nb_ligne-1][nb_colonnes-2]   # valeur de Xn qui nous permet de résoudre le reste du système
    liste_solutions.append(v)
    for i in range(2,nb_ligne+1):
        n = Taug[nb_ligne-i][nb_colonnes-1]
        for j in range(2,nb_colonnes+1):
            if ((nb_colonnes - j) > (nb_ligne - i)):
                #if abs(Taug[nb_ligne-i , nb_colonnes-j] - 0) <= 10**-10:
                n = n - Taug[nb_ligne-i , nb_colonnes-j]*liste_solutions[comptage]  #on bascule toutes les valeurs de l'autre coté sauf le X qu'on cherche à calculer.
                comptage+=1
        n = n / Taug[nb_ligne-i, nb_colonnes-i-1]  #on divise et on obtient la valeur d'une des inconnu du système
        v=n
        comptage = 0
        liste_solutions.append(v)
        X=np.asarray(liste_solutions)
    return X


def ReductionGauss(Aaug):

    '''
    Cette fonction permet d'otenir un matrice triangulaire supérieur par la méthode
    du pivot de gauss.
    Elle retourne cette matrice et prend en argument la matrice augmenté de A et B.
    '''

    global nb_ligne, nb_colonnes
    pivot = 0
    for k in range(0,nb_colonnes):
        for i in range(k+1,nb_ligne):
            pivot = (Aaug[i,k])/(Aaug[k,k])
            Aaug[i,:] = Aaug[i,:] - pivot * Aaug[k,:] 
    return (Aaug)

Y = list()
ERREUR_Gauss=list()
def Gauss(A,B):

    '''
    Cette fonction regroupe les fonctions nécéssaires pour effectuer la méthode du pivot de gauss pour trouver les solution d'un système à 
    l'aide de matrices.
    Elle retourne le temps de calcul nécéssaire pour effectuer cette méthode.
    '''

    global nb_ligne, nb_colonnes
    T1 = t.time()
    Aaug=np.c_[A,B]
    nb_ligne, nb_colonnes = np.shape(Aaug)
    Taug = ReductionGauss(Aaug)
    X_gauss = ResolutionSystTriSup(Taug)
    T2 = t.time()
    tps_calcul = T2-T1
    erreur = np.linalg.norm(A.dot(X_gauss) - np.ravel(B))
    ERREUR_Gauss.append(erreur)
    Y.append(tps_calcul)
    return tps_calcul


###################################  Décomposition LU ##################################

def DecompositionLU(A):

    '''
    Grace à cette fonction on peut obtenir la décomposition LU d'une matrice carrée A tel que A=LU.
    '''

    global nb_ligne, nb_colonnes
    nb_ligne, nb_colonnes = np.shape(A)
    #création d'une matrice L vièrge
    n = nb_ligne

    L = np.random.rand(n, n)
    for k in range(0,n):
        for i in range(0,n):
            L[i,k] = 0

    #Set up pour faire une réduction de la matrice carrée A et réduction.
    #à chaque création du pivot il est ajouté à la matrice L
    pivot = 0

    for k in range(0,nb_colonnes):
        for i in range(k+1,nb_ligne):
            pivot = (A[i,k])/(A[k,k])
            L[i,i] = 1
            L[i,k] = pivot
            A[i,:] = A[i,:] - pivot * A[k,:] 
    L[0,0] = 1

    
    #print des résultats pour les vérifier, la matrice A est devenue la matrice U
    U = A

    '''
    print("la fonction U : \n")
    print(U)
    print("la fonction L : \n")
    print(L)
    '''

    return (U,L)

def ResolutionSystTriSup_U(U,Y):

    '''
    Cette fonction permet de résoudre un système à partir d'une matrice triangulaire supérieur (matrice U)..
    Elle est légèrement différentes de ResolutionSystTriSup() car elle prend en argument U et une matrice Y contenant les solutions de LY=B
    '''

    global nb_ligne, nb_colonnes
    comptage=0
    Taug = np.c_[U,Y]
    nb_ligne, nb_colonnes = np.shape(Taug)
    liste_solutions = []
    v = Taug[nb_ligne-1][nb_colonnes-1]/Taug[nb_ligne-1][nb_colonnes-2]   # valeur de Xn qui nous permet de résoudre le reste du système
    liste_solutions.append(v)
    for i in range(2,nb_ligne+1):
        n = Taug[nb_ligne-i][nb_colonnes-1]
        for j in range(2,nb_colonnes+1):
            if ((nb_colonnes - j) > (nb_ligne - i)):
                #if abs(Taug[nb_ligne-i , nb_colonnes-j] - 0) <= 10**-10:
                n = n - Taug[nb_ligne-i , nb_colonnes-j]*liste_solutions[comptage]  #on bascule toutes les valeurs de l'autre coté sauf le X qu'on cherche à calculer.
                comptage+=1
        n = n / Taug[nb_ligne-i, nb_colonnes-i-1]  #on divise et on obtient la valeur d'une des inconnu du système
        v=n
        comptage = 0
        liste_solutions.append(v)
    return liste_solutions

def ResolutionSystTriginf(L,B):

    '''
    Cette fonction permet de résoudre un système à partir d'une matrice triangulaire inférieur (matrice L).
    Elle prend en argument L et B et retourne les solutions sous forme d'une matrice.
    '''

    Taug=np.c_[L,B]
    nb_ligne, nb_colonnes = np.shape(Taug)
    comptage=0
    Y = []
    v = Taug[0][nb_colonnes-1]/Taug[0][0]   # valeur de Xn qui nous permet de résoudre le reste du système
    Y.append(v)
    for i in range(1,nb_ligne):
        n = Taug[i][nb_colonnes-1]
        for j in range(0,nb_colonnes-1):
            if (j < i):
                #print("i :", i, " j :", j)
                #print("djdjd  : ", Taug[i , j])
                #if abs(Taug[i , j] - 0) <= 10**-10:
                n = n - Taug[i , j]*Y[comptage]  #on bascule toutes les valeurs de l'autre coté sauf le X qu'on cherche à calculer.
                comptage+=1
        n = n / Taug[i,i]  #on divise et on obtient la valeur d'une des inconnu du système
        v=n       
        comptage = 0
        Y.append(v)
    Y = np.asarray(Y)
    Y = Y.T
    return Y

Y_LU = list()
ERREUR_LU=list()
def resolution_LU(A,B):

    '''
    Cette fonction regroupe les fonctions nécéssaires pour effectuer la méthode LU pour trouver les solution d'un système à 
    l'aide de matrices.
    Elle retourne le temps de calcul nécéssaire pour effectuer cette méthode.
    '''

    T1 = t.time()
    U,L = DecompositionLU(A)
    #### résolution de LY=B ####
    Y = ResolutionSystTriginf(L,B)
    #### Résolution UX=Y ####
    X = ResolutionSystTriSup_U(U,Y)
    X = np.asarray(X)
    erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
    ERREUR_LU.append(erreur)
    T2 = t.time()
    tps_calcul = T2-T1
    Y_LU.append(tps_calcul)
    return X


######################################## Pivot partiel ###############################

def GaussChoixPivotPartiel(A,B):

    '''
    Cette fonction permet d'otenir un matrice triangulaire supérieur par la méthode
    du pivot partiel.
    Elle retourn cette matrice augmenté et prend en argument deux matrices A,B.
    '''

    Aaug=np.c_[A,B]
    nb_ligne, nb_colonnes = np.shape(Aaug)
    #réduction de Gauss
    L_echange=0
    pivot = 0
    for k in range(0,nb_colonnes):
        for i in range(k+1,nb_ligne):
            #exception si le pivot est nul
            if Aaug[k,k] == 0:
                plusgrand = 0
                for l in range (i, nb_ligne):
                    #selection du plu grand module dans la ligne
                    if abs(Aaug[l,k]) > plusgrand:
                        plusgrand = abs(Aaug[l,k])
                        L_echange = l
                #échange des lignes
                c=Aaug[i-1,:].copy()
                Aaug[i-1,:] = Aaug[L_echange,:]
                Aaug[L_echange,:]= c
            #utilisation du pivot pour trianguler 
            pivot = (Aaug[i,k])/(Aaug[k,k])

            Aaug[i,:] = Aaug[i,:] - pivot * Aaug[k,:]
 
    return Aaug 

Y_PP = list()
ERREUR_PP=list()
def pivot_partiel(A,B):

    '''
    Cette fonction regroupe les fonctions nécéssaires pour effectuer la méthode pivot partiel pour trouver les solution d'un système à 
    l'aide de matrices.
    Elle retourne le temps de calcul nécéssaire pour effectuer cette méthode.
    '''

    T1=t.time()
    Aaug = GaussChoixPivotPartiel(A,B)
    X_PP = ResolutionSystTriSup(Aaug)
    T2=t.time()
    tps_calcul=T2-T1
    Y_PP.append(tps_calcul)
    erreur = np.linalg.norm(A.dot(X_PP) - np.ravel(B))
    ERREUR_PP.append(erreur)
    return tps_calcul




################################### Pivot Total #######################################

def GaussChoixPivotTotal(A,B):

    '''
    Cette fonction permet d'otenir un matrice triangulaire supérieur par la méthode
    du pivot total.
    Elle retourn cette matrice augmenté et prend en argument deux matrices A,B.
    '''

    Aaug=np.c_[A,B]
    nb_ligne, nb_colonnes = np.shape(Aaug)
    #réduction de Gauss
    L_echange=0
    pivot = 0
    plusgrand = 0
    for k in range(0,nb_colonnes):
        for i in range(k+1,nb_ligne):
            if Aaug[k,k] == 0:
                print('passage')
                for l in range (i, nb_ligne):
                    if abs(Aaug[l,k]) > plusgrand:
                        print(Aaug[l,k])
                        plusgrand = abs(Aaug[l,k])
                        L_echange = l
                        a=0
                for c in range (i, nb_colonnes):
                    if abs(Aaug[k,c]) > plusgrand:
                        a=1
                        plusgrand = abs(Aaug[k,c])
                        L_echange = c

                if a==0:
                    c=Aaug[i-1,:].copy()
                    Aaug[i-1,:] = Aaug[L_echange,:]
                    Aaug[L_echange,:]= c
                    
                else:
                    c=Aaug[:,i-1].copy()
                    Aaug[:,i-1] = Aaug[:,L_echange]
                    Aaug[:,L_echange]= c
            
            pivot = (Aaug[i,k])/(Aaug[k,k])
            Aaug[i,:] = Aaug[i,:] - pivot * Aaug[k,:]
            
    return Aaug 

Y_PT=list()
ERREUR_PT=list()
def pivotTotal(A,B):

    '''
    Cette fonction regroupe les fonctions nécéssaires pour effectuer la méthode pivot total pour trouver les solution d'un système à 
    l'aide de matrices.
    Elle retourne le temps de calcul nécéssaire pour effectuer cette méthode.
    '''

    T1=t.time()
    Aaug = GaussChoixPivotTotal(A,B)
    X_PT = ResolutionSystTriSup(Aaug)
    T2=t.time()
    tps_calcul=T2-T1
    Y_PT.append(tps_calcul)
    erreur = np.linalg.norm(A.dot(X_PT) - np.ravel(B))
    ERREUR_PT.append(erreur)
    return tps_calcul

################################ Linalg.solve ################################
Y_linalg = list()
ERREUR_Linalg=list()
def linalg_solve(A,B):

    '''
    Cette fonction permet de résoudre un système AX=B par la méthode linalg.solve présente de base sur python
    Elle retourne le temps de calcul nécéssaire pour effectuer cette méthode.
    '''

    T1=t.time()
    X = np.linalg.solve(A,B)
    T2=t.time()
    tps_calcul = T2-T1 
    Y_linalg.append(tps_calcul)
    erreur = np.linalg.norm(A.dot(X) - np.ravel(B))
    ERREUR_Linalg.append(erreur)


###########################################################################################
#                              Création des différents graphiques                         #
###########################################################################################
for i in range(100, 1500, 100):
    X.append(i)
    A, B = random_mat(i)
    Gauss(A,B)
    resolution_LU(A,B)
    linalg_solve(A,B)
    pivot_partiel(A,B)
    pivotTotal(A,B)


##### Courbe temps de fonction de la taille de la matrice #####
plt.ylabel("temps en seconde")
plt.xlabel(" n ")
plt.title("Courbe du temps d'exécution en fonction de la taille n de la matrice")
plt.plot(X,Y, label='Pivot Gauss')
plt.plot(X,Y_LU, label='LU')
plt.plot(X,Y_linalg, label='linalg.solve')
plt.plot(X,Y_PP, label='Pivot partiel')
plt.plot(X,Y_PT, label='Pivot total')
plt.grid()
plt.legend()
plt.show()

###### Courbe Logarithmique ######
plt.loglog(X,Y, label='courbe Ln de Pivot Gauss')
plt.loglog(X,Y_LU, label='courbe Ln de LU')
#plt.loglog(X,Y_linalg, label='Courbe Ln de linalg.solve')
plt.loglog(X,Y_PP, label='courbe ln de pivot partiel')
plt.loglog(X,Y_PT, label='courbe ln de pivot total')
plt.grid()
plt.ylabel("temps en seconde")
plt.xlabel(" n ")
plt.title("Courbe logarithmique ")
plt.legend()
plt.show()

##### Courbe des erreurs #####
plt.plot(X,ERREUR_Gauss, label='Pivot Gauss')
plt.plot(X,ERREUR_LU, label='LU')
plt.plot(X,ERREUR_Linalg, label='linalg.solve')
plt.plot(X,ERREUR_PT, label='Pivot total')
plt.plot(X,ERREUR_PP, label='Pivot partiel')
plt.grid()
plt.legend()
plt.ylabel("Erreur ||=AX-B||")
plt.xlabel(" n ")
plt.title("Courbe de l'erreur en fonction de la taille n de la matrice")
   
plt.show()

###### Courbe log des erreurs  #####
plt.semilogy(X,ERREUR_Gauss, label='Pivot Gauss')
plt.semilogy(X,ERREUR_LU, label='LU')
plt.semilogy(X,ERREUR_Linalg, label='linalg.solve')
plt.semilogy(X, ERREUR_PP, label='Pivot partiel')
plt.semilogy(X, ERREUR_PT, label='Pivot total')
plt.legend()
plt.grid()
plt.ylabel("Erreur ||=AX-B||")
plt.xlabel(" n ")
plt.title("Courbe du log de l'erreur en fonction de la taille n de la matrice")
plt.show()

print(ERREUR_PP)
print(ERREUR_PT)