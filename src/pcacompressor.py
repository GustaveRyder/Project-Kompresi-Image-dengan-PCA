import cv2
import numpy as np 

def compress_image(img):

    #ngetung rata2
    M = np.mean(img.T, axis=1)
    #ngurangi karo mean e
    C = img - M
    #ngitung covariance
    V = np.cov(C.T)
    #ngitung eigen value, ro eigen vectors
    values, vectors =np.linalg.eig(V)

    p = np.size(vectors, axis =1)
    #ngurutke eigen values escending
    idx = np.argsort(values)
    idx = idx [::-1]
    #sorting eigen vectors
    vectors = vectors [:,idx]
    values = values [idx]

    num_PC = 100 #isi sak sak e nek 0 soyo bruwet

    if num_PC < p or num_PC > 0:
        vector = vectors[:, range(num_PC)]
    #recontruksi gambar
    score = np.dot(C, vectors)
    constructed_img = np.dot(score, vectors.T) + M 
    constructed_img = np.uint8(np.absolute(constructed_img))
    return constructed_img


#lokasi nyang folder sik podo ro code py ne
imgpath ="wp1827100.jpg"
img_array = cv2.imread(imgpath)

#mbagi dadi 3 (biru, ijo, abang)
b, g, r = cv2.split(img_array)

#proses kompres 
b_compressed = compress_image(b)
g_compressed = compress_image(g)
r_compressed = compress_image(r)

#biru, ijo, abang digabung
reconstructed_img = cv2.merge((b_compressed, g_compressed, r_compressed))


#output file
cv2.imwrite("Compressed_image.jpg", reconstructed_img)
