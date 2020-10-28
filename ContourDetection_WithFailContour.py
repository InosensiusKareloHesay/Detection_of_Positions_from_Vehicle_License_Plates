import cv2
import os
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def findDirectory(directory, folder, folderJarak, folderTinggi):
    count = directory.count('\\')
    listDir = directory.split('\\')
    newDirectory = ""
    for i in range(count):
        if (i == (count - 1)):
            newDirectory += listDir[i]
            break;
        else:
            newDirectory += listDir[i] + '\\'
    dirHasil = os.path.join(newDirectory, "HASIL")
    if not os.path.exists(dirHasil):
        os.mkdir(dirHasil)
    dirWaktu = os.path.join(dirHasil, folder)
    if not os.path.exists(dirWaktu):
        os.mkdir(dirWaktu)
    dirJarak = os.path.join(dirWaktu, folderJarak)
    if not os.path.exists(dirJarak):
        os.mkdir(dirJarak)
    dirTinggi = os.path.join(dirJarak, folderTinggi)
    if not os.path.exists(dirTinggi):
        os.mkdir(dirTinggi)
    return dirTinggi

def saveCrop(directory, file_name, image):
    dirCrop = os.path.join(directory, "Cropping")
    if not os.path.exists(dirCrop):
        os.mkdir(dirCrop)
    cv2.imwrite(os.path.join(dirCrop, file_name), image)

def saveContour(directory, file_name, image):
    dirContour = os.path.join(directory, "Contour")
    if not os.path.exists(dirContour):
        os.mkdir(dirContour)
    cv2.imwrite(os.path.join(dirContour, file_name), image)

def saveAlign(directory, file_name, image):
    dirAlign = os.path.join(directory, "Align")
    if not os.path.exists(dirAlign):
        os.mkdir(dirAlign)
    cv2.imwrite(os.path.join(dirAlign, file_name), image)

def saveContourGagal(directory, file_name, image):
    dirCon = os.path.join(directory, "ContourGagal")
    if not os.path.exists(dirCon):
        os.mkdir(dirCon)
    cv2.imwrite(os.path.join(dirCon, file_name), image)

def Binerr(directory, file_name, image):
    binerr = os.path.join(directory, "Binerr")
    if not os.path.exists(binerr):
        os.mkdir(binerr)
    cv2.imwrite(os.path.join(binerr, file_name), image)

def saveContourMerah(directory, file_name, image):
    dirCon = os.path.join(directory, "Merah")
    if not os.path.exists(dirCon):
        os.mkdir(dirCon)
    cv2.imwrite(os.path.join(dirCon, file_name), image)

def RUN(self):
    directory = input('Masukkan directory folder : ')
    totalBerhasil = 0
    totalFoto = 0

    for folder in os.listdir(directory):  # pagi - sore
        new_direc = directory + '//' + folder
        for folderJarak in os.listdir(new_direc):  # 2,3,4 meter
            new_direcJarak = new_direc + '//' + folderJarak
            for folderTinggi in os.listdir(new_direcJarak):  # 1,2,3,4 meter
                new_direcTinggi = new_direcJarak + '//' + folderTinggi

                tidakTerdeteksi = 0
                tidakAlign = 0

                for file_name in sorted(os.listdir(new_direcTinggi)):
                    image = cv2.imread(os.path.join(new_direcTinggi, file_name))
                    image = cv2.resize(image, (300, 150))
                    imageAwal = image.copy()
                    directoryTinggi = findDirectory(directory, folder, folderJarak, folderTinggi)

                    # ------------------ Proses Preprocessing Image --------------

                    # convert image menjadi blur menggunakan gaussian smoothing
                    blurred = cv2.GaussianBlur(image, (5, 5), 0.5)

                    # convert image menjadi grayscale
                    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

                    # Image binerisasi menggunakan adaptive thresholding
                    treshold = cv2.adaptiveThreshold(gray, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,17, 8)
                    Binerr(directoryTinggi, file_name, treshold)
                    # Morfologi image menggunakan dilasi
                    treshold = cv2.dilate(treshold, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

                    # ------------------ Proses Contour Detection -----------------------

                    # Mencari contour
                    contours, hirarki = cv2.findContours(treshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    areaList = []

                    # mencari contour paling luas
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        areaList.append(area)

                    contBesar = max(areaList)
                    indexBesar = areaList.index(contBesar)

                    # Menggambar contour asli
                    cv2.drawContours(image, [contours[indexBesar]], -1, (30, 144, 255), 2)

                    # Perbaikan contour pertama menggunakan contour approximation
                    perimeter = cv2.arcLength(contours[indexBesar], True)
                    alignContour = cv2.approxPolyDP(contours[indexBesar], 0.06 * perimeter, True)

                    # Menggambar hasil contour setelah perbaikan pertama (merah)
                    cv2.drawContours(image, [alignContour], -1, (0, 0, 255), 2)
                    saveContourMerah(directoryTinggi, file_name, image)
                    # Perbaikan contour akhir menggunakan seleksi kondisi tertentu
                    x, y, w, h = cv2.boundingRect(alignContour)
                    treshold2 = treshold.copy()[y:y - 3 + h + 6, x:x - 3 + w + 6]
                    tresholdCrop = cv2.resize(treshold2, (300, 150))

                    if len(alignContour) == 2:
                        if (alignContour[0][0][1] < alignContour[1][0][1] and alignContour[0][0][0] <
                                alignContour[1][0][0]):
                            xSatu = alignContour[0][0][0]
                            xDua = alignContour[1][0][0]
                            i = 0
                            while tresholdCrop[0][i] != 255 and tresholdCrop[1][i] != 255 and xSatu < 50:
                                xSatu += 1
                                i += 1
                            j = 0
                            maxDua = xDua - 20
                            while tresholdCrop[0][j] != 255 and xDua > maxDua:
                                xDua -= 1
                                j += 1
                            k = 0
                            alignContourBaru = np.array(
                                [[[xSatu - 3, alignContour[0][0][1]]], [[xDua, alignContour[0][0][1]]],
                                 [[alignContour[1][0][0], alignContour[1][0][1]]],
                                 [[xSatu + j - 3, alignContour[1][0][1]]]])

                        elif (alignContour[0][0][1] < alignContour[1][0][1] and alignContour[0][0][0] >
                              alignContour[1][0][0]):
                            xTiga = alignContour[0][0][0]
                            i = 299
                            while tresholdCrop[149][i] != 255 and tresholdCrop[148][i] != 255 and xTiga > 250:
                                xTiga -= 1
                                i -= 1
                            alignContourBaru = np.array([[[alignContour[1][0][0] + (299 - i), alignContour[0][0][1]]],
                                                         [[alignContour[0][0][0], alignContour[0][0][1]]],
                                                         [[xTiga, alignContour[1][0][1]]],
                                                         [[alignContour[1][0][0], alignContour[1][0][1]]]])

                        else:
                            xTiga = alignContour[1][0][0]
                            i = 299
                            while tresholdCrop[149][i] != 255 and tresholdCrop[148][i] != 255 and xTiga > 250:
                                xTiga -= 1
                                i -= 1
                            alignContourBaru = np.array([[[alignContour[0][0][0] + (299 - i), alignContour[1][0][1]]],
                                                         [[alignContour[1][0][0], alignContour[1][0][1]]],
                                                         [[xTiga, alignContour[0][0][1]]],
                                                         [[alignContour[0][0][0], alignContour[0][0][1]]]])

                        #                         if(cv2.contourArea(alignContour) == cv2.contourArea(alignContourBaru)) :
                        #                             # Menggambar hasil contour setelah perbaikan akhir
                        #                             cv2.drawContours(image, [alignContour], -1, (255,0,0), 2)
                        #                         else:
                        alignContour = alignContourBaru

                        # Menggambar hasil contour setelah perbaikan akhir (hijau)
                        cv2.drawContours(image, [alignContour], -1, (0, 255, 0), 2)

                    # Seleksi contour yang ukurannya tidak sesuai
                    x, y, w, h = cv2.boundingRect(alignContour)
                    if (x < 0 or y < 0): x, y = 0, 0
                    ras = format(w / h, '.2f')
                    if 2 > float(ras) or len(alignContour) != 4:
                        saveContourGagal(directoryTinggi, file_name, image)
                        tidakTerdeteksi += 1
                        continue

                    # Jika hasil contour bukan segi empat, maka proses align tidak bisa dilakukan
                    #                     if len(alignContour) != 4:
                    #                         saveContourGagal(directoryTinggi, file_name, image)
                    #                         tidakAlign += 1
                    #                         continue

                    # Menyimpan contour yang sesuai untuk proses aligning
                    saveContour(directoryTinggi, file_name, image)

                    # cropping image sesuai contour
                    hasilCrop = imageAwal[y:y - 3 + h + 6, x:x - 3 + w + 6]

                    # Menyimpan hasil cropping image
                    saveCrop(directoryTinggi, file_name, hasilCrop)

                    # ------------------ Proses Aligning Image --------------
                    hasilAlign = four_point_transform(imageAwal, alignContour.reshape(4, 2))
                    saveAlign(directoryTinggi, file_name, hasilAlign)

                # ------------------ Perhitungan Akurasi
                total = len(os.listdir(new_direcTinggi))
                gagal = tidakTerdeteksi + tidakAlign
                berhasil = total - gagal
                akurasi = (berhasil / total) * 100
                totalFoto += total
                print("=== Hasil dari %s Hari, dengan %s dan %s :" % (folder, folderJarak, folderTinggi))
                print("Tidak Terdeteksi : " + str(tidakTerdeteksi))
                print("Tidak Bisa Align : " + str(tidakAlign))

                print("\nTotal Foto : " + str(total))
                print("Total Gagal : " + str(gagal))
                print("Total Berhasil : " + str(berhasil))
                print("Akurasi Sementara : " + str(akurasi) + " %\n")

                # hasilSalah = input('Masukkan jumlah hasil yang salah : ')
                hasilSalah = 0
                gagal += int(hasilSalah)
                berhasil = total - gagal
                totalBerhasil += berhasil
                akurasi = (berhasil / total) * 100

                print("Total Gagal : " + str(gagal))
                print("Total Berhasil : " + str(berhasil))
                print("\nAkurasi Akhir: " + str(akurasi) + " %")

    print("\n==============HASIL KESELURUHAN==========================")
    print("\nTotal Foto : " + str(totalFoto))
    print("Total Berhasil : " + str(totalBerhasil))
    print("Total Gagal : " + str(totalFoto - totalBerhasil))
    print("\nAkurasi Akhir: " + str((totalBerhasil / totalFoto) * 100) + " %")


if __name__ == '__main__':
    RUN('self')

# ##
