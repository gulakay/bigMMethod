import numpy as np
import warnings

def simplex(tür, A, B, C, D, M):
    """Doğrusal programlama modeli için optimal noktayı hesaplar: A*x <= B, Z = C' * x optimizasyonu

    Argümanlar:
    tür -- optimizasyon türü, 'max' veya 'min' olabilir
    A   -- modelin A matrisi (numpy array)
    B   -- modelin B matrisi, sütun vektörü (numpy array)
    C   -- modelin C matrisi, sütun vektörü (numpy array)
    D   -- modelin kısıtlama tiplerinin olduğu sütun vektörü (numpy array), 1 <=, 0 =, -1 >=
            <= kısıtlamaları için bir şey yapma
            = kısıtlamaları için bir yapay değişken ve amaç fonksiyonuna büyük M ekle (min --> +M, max --> -M)
            >= kısıtlamalarını -1 ile çarp
    M   -- büyük M değeri
    """

    # m -- kısıtlama sayısı
    # n -- değişken sayısı
    (m, n) = A.shape

    temel_değişkenler = []
    say = n

    # yeni değişkenlerle matris
    R = np.eye(m)

    # yeni değişkenlerin değerleri
    P = B

    # yapay değişkenlerin pozisyon göstergesi
    yapay = []

    for i in range(m):
        if D[i] == 1:
            # artıklık değişkenini amaç fonksiyonuna ekle
            C = np.vstack((C, [[0]]))

            # artıklık değişkenini temel değişken olarak kaydet
            say = say + 1
            temel_değişkenler = temel_değişkenler + [say-1]

            yapay.append(0)

        elif D[i] == 0:
            # amaç fonksiyonuna yapay değişkeni ve büyük M değerini ekle
            if tür == 'min':
                C = np.vstack((C, [[M]]))
            else:
                C = np.vstack((C, [[-M]]))

            # yapay değişkeni temel değişken olarak kaydet
            say = say + 1
            temel_değişkenler = temel_değişkenler + [say-1]

            yapay.append(1)
        elif D[i] == -1:
            # artık ve yapay değişkenleri amaç fonksiyonuna ekle
            if tür == 'min':
                C = np.vstack((C, [[0], [M]]))
            else:
                C = np.vstack((C, [[0], [-M]]))

            R = repeatColumnNegative(R, say + 1 - n)
            P = insertZeroToCol(P, say + 1 - n)

            # yapay değişkeni temel değişken olarak kaydet
            say = say + 2
            temel_değişkenler = temel_değişkenler + [say-1]

            yapay.append(0)
            yapay.append(1)
        else:
            print("geçersiz durum")

    # mevcut köşe noktası
    X = np.vstack((np.zeros((n, 1)), P))

   
    # matris A'ya yeni değişkenleri ekle
    A = np.hstack((A, R))

    # simplex tablosu
    st = np.vstack((np.hstack((-np.transpose(C), np.array([[0]]))), np.hstack((A, B))))

    # sütun sayısı
    (rows, cols) = st.shape

    # temel_değişkenler = ((n + 1):n+m)'

    print('\nsimplex tablosu\n')
    print(st)
    print('\nmevcut temel değişkenler\n')
    print(temel_değişkenler)
    print('\noptimal nokta\n')
    print(X)

    # z != 0 olduğunu kontrol et (yapay değişkenler olduğunda)
    z_optimal = np.matmul(np.transpose(C), X)

    print('\nmevcut Z\n\n', z_optimal)

    if z_optimal != 0:
        for i in range(m):
            if D[i] == 0 or D[i] == -1:
                if tür == 'min':
                    st[0, :] = st[0, :] + M * st[1+i, :]
                else:
                    st[0, :] = st[0, :] - M * st[1+i, :]

        print('\ndüzeltilecek simplex tablosu\n')
        print(st)

    iterasyon = 0
    while True:
        if tür == 'min':
            # en pozitif değeri seç
            w = np.amax(st[0, 0:cols-1])
            iw = np.argmax(st[0, 0:cols-1])
        else:
            # en negatif değeri seç
            w = np.amin(st[0, 0:cols-1])
            iw = np.argmin(st[0, 0:cols-1])

        if w <= 0 and tür == 'min':
            print('\nGlobal optimum nokta\n')
            break
        elif w >= 0 and tür == 'max':
            print('\nGlobal optimum nokta\n')
            break
        else:
            iterasyon = iterasyon + 1

            print('\n----------------- İterasyon {} -------------------\n'.format(iterasyon))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T = st[1:rows, cols-1] / st[1: rows, iw]

            R = np.logical_and(T != np.inf, T > 0)
            (k, ik) = minWithMask(T, R)

            # mevcut z satırı
            cz = st[[0], :]

            # pivot elemanı
            pivot = st[ik+1, iw]

            # pivot satırı pivot elemanına bölünmüş
            prow = st[ik+1, :] / pivot

            st = st - st[:, [iw]] * prow

            # pivot satırı özel bir durumdur
            st[ik+1, :] = prow

            # yeni temel değişken
            temel_değişkenler[ik] = iw

            print('\nmevcut temel değişkenler\n')
            print(temel_değişkenler)

            # yeni köşe noktası
            basic = st[:, cols-1]
            X = np.zeros((say, 1))

            t = np.size(temel_değişkenler)

            for k in range(t):
                X[temel_değişkenler[k]] = basic[k+1]

            print('\nmevcut optimal nokta\n')
            print(X)

            # yeni z değeri
            C = -np.transpose(cz[[0], 0:say])

            z_optimal = cz[0, cols-1] + np.matmul(np.transpose(C), X)
            st[0, cols-1] = z_optimal

            print('\nsimplex tablosu\n\n')
            print(st)

            print('\nmevcut Z\n\n')
            print(z_optimal)

    # pozitif bir yapay değişken olduğunu kontrol et (çözüm yok)
    tv = np.size(yapay)
    for i in range(tv):
        if yapay[i] == 1:
            if X[n + i] > 0:
                print('\nçözüm yok\n')
                break

    return (z_optimal[0, 0], X)


def minWithMask(x, maske):
    min = 0
    imin = 0

    n = np.size(x)

    for i in range(n):
        if maske[i] == 1:
            if min == 0:
                min = x[i]
                imin = i
            else:
                if min > x[i]:
                    min = x[i]
                    imin = i

    return (min, imin)


def repeatColumnNegative(Mat, h):
    """Sütunu h ile çarpılmış -1 ile tekrarlayın"""
    (r, c) = Mat.shape
    Mat = np.hstack((Mat[:, 0:h-1], -Mat[:, [h-1]], Mat[:, h-1:c]))

    return Mat


def insertZeroToCol(col, h):
    """Sütuna sıfır ekle"""
    k = np.size(col)
    col = np.vstack((col[0:h-1, [0]], np.array([[0]]), col[h-1:k, [0]]))

    return col


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

#ÖRNEK 4
'''Maximize Z = 3x1 + 4x2 + x3
Subject to:
x1 + x2 + 2x3 <= 8
x1 + x2 + x3 <= 7'''

'''(z, x) = simplex('max', 
                 np.array([[1, 1, 2], [1, 1, 1]]),
                 np.array([[8], [7]]),
                 np.array([[3], [4], [1]]),
                 np.array([[-1], [-1]]),
                 100)'''

#ÖRNEK 5
'''Maximize Z = x1 + 2x2 + 4x3
Subject to:
x1 - x2 >= 0
x1 + x2 + x3 <= 6
2x1 + x2 + 2x3 <= 4
x1, x2, x3 >= 0'''

'''(z, x) = simplex('max', 
                 np.array([[1, -1, 0], [1, 1, 1], [2, 1, 2]]),
                 np.array([[0], [6], [4]]),
                 np.array([[1], [2], [4]]),
                 np.array([[-1], [1], [1]]),
                 100)'''

#ÖRNEK 6
'''Maximize Z = x1 + 5x2
Subject to:
x1 + x2 >= 9
3x1 + x2 >= 21
x1 + 4x2 <= 24
x1, x2 >= 0'''

(z, x) = simplex('max', 
                 np.array([[1, 5], [1, 1], [3, 1], [1, 4]]),
                 np.array([[0], [9], [21], [24]]),
                 np.array([[1], [5]]),
                 np.array([[-1], [-1], [-1], [1]]),
                 100)

#ÖRNEK 7
'''Minimize Z = -6x1 - 8x2
Subject to:
4x1 + 5x2 <= 40
4x1 + 10x2 <= 60
x1, x2 >= 0'''

'''(z, x) = simplex('min', 
                 np.array([[-6, -8]]),
                 np.array([[0]]),
                 np.array([[-6], [-8]]),
                 np.array([[-1], [-1]]),
                 100)'''



