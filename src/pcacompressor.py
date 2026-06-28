import cv2
import numpy as np 

def compress_image(img, num_PC):

    #ngetung rata2
    M = np.mean(img.T, axis=1)
    #ngurangi karo mean e
    C = img - M
    #ngitung covariance
    V = np.cov(C.T)
    #ngitung eigen value, ro eigen vectors
    values, vectors =np.linalg.eig(V)

    p = np.size(vectors, axis=1)
    #ngurutke eigen values escending
    idx = np.argsort(values)
    idx = idx [::-1]
    #sorting eigen vectors
    vectors = vectors [:,idx]
    values = values [idx]

    vector = vectors[:, range(num_PC)]
    #recontruksi gambar
    score = np.dot(C, vector)
    constructed_img = np.dot(score, vector.T) + M 
    constructed_img = np.clip(constructed_img.real, 0, 255).astype(np.uint8)
    return constructed_img


#lokasi nyang folder sik podo ro code py ne
imgpath ="wp1827100.jpg"
img_array = cv2.imread(imgpath)
tingkat_kompresi = 50

#mbagi dadi 3 (biru, ijo, abang)
b, g, r = cv2.split(img_array)

#proses kompres 
b_compressed = compress_image(b, tingkat_kompresi)
g_compressed = compress_image(g, tingkat_kompresi)
r_compressed = compress_image(r, tingkat_kompresi)

#biru, ijo, abang digabung
reconstructed_img = cv2.merge((b_compressed, g_compressed, r_compressed))

#output file
cv2.imwrite("Compressed_image.jpg", reconstructed_img)
