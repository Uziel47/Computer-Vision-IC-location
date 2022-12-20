import cv2
from matplotlib import pyplot as plt
import numpy as np
#import scipy.fftpack as fft
from scipy import ndimage as nd
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#PARA VISUALIZAR EL COMPONENTE SELECCIONADO CON COLOR
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.measure import label


def binview(img, mask, color='r', dilate_pixels=1):
    """
    Displays a gray or color image 'img' overlaid by color pixels determined a by binary image 'mask'. It is useful to
    display the edges of an image.

    Args:
        img: gray scale image (X-ray)
        mask: binary image that works as mask
        color: string to define pixel color.
                'r': red (default)
                'g': green
                'b': blue
                'y': yellow
                'c': cyan
                'm': magenta
                'k': black
                'w': white

        dilate_pixels (int): Number of pixels used for dilate the mask.

    Returns:
        img_color (ndarray): output image with a mask overlaid.
    """

    # Defines colors

    colors = {
        'r': np.array([1, 0, 0]),
        'g': np.array([0, 1, 0]),
        'b': np.array([0, 0, 1]),
        'y': np.array([1, 1, 0]),
        'c': np.array([0, 1, 1]),
        'm': np.array([1, 0, 1]),
        'k': np.array([0, 0, 0]),
        'w': np.array([1, 1, 1])
    }
    # Ensure do not modify the original color image and the mask
    img_color = img.copy()

    mask_ = mask.copy()
    mask_ = binary_dilation(mask_, np.ones((dilate_pixels, dilate_pixels)))

    # Defines the pixel color used for the mask in the figure.
    cc = colors[color]

    # remove artifacts connected to image border
    cleared = clear_border(mask_)
    if np.all(cleared):
        mask_ = cleared

    # label image regions
    label_image = label(mask_)
    img_color = label2rgb(label_image, image=img_color, colors=[cc], bg_label=0)

    return img_color  # add(img_color, img_color)

video = cv2.VideoCapture(0)

#EVENTOS ASCII
asci_q = ord('q')
asci_s = ord('s')
asci_w = ord('w')

#FILTRO GAUSSIANO PASA ALTA PARA TRANSFORMADA 
#np.arange(inicio,parada,paso)
F1=np.arange(-320,320,1) #375+375 nos da el tamaño en X del recorte
F2=np.arange(-240,240,1)   #lo mismo pero en el eje Y
[X,Y]=np.meshgrid(F1,F2)
R=np.sqrt(X**2+Y**2)
R=R/np.max(R)
sigma = 0.1
# Filtro pasa altas con función gaussiana
filt_gauss = 1-np.exp(-(R**2)/(2*sigma**2))

#######SOLO VISUALIZACIÓN DEL FILTRO GAUSS
# # Graficamos el filtro en 2D
# plt.figure()
# plt.imshow(filt_gauss)
# plt.show()

# # Graficamos el filtro en 3D
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(X, Y, filt_gauss,cmap=cm.viridis)
# plt.show()

color = (0,255,0)
grosor = 2

#RECORTE DEL FRAME DONDE SE VA A PROCESAR EL IC
x1,x2 = 120,544
y1,y2 = 72,394

#MATRIX
kernel = np.ones((5,5),np.uint8)

while(True):
    
    ret, frame = video.read()
    key = cv2.waitKey(1)    
    if key & 0xFF == asci_q:
        print("PROGRAMA FINALIZADO POR TECLADO")
        break
    if key & 0xFF == asci_s:
        print("COMPONENTE SELECCIONADO")
        
        frame_copy = frame.copy()
    
        # #ESCALA DE GRISES Y SAVISADO PARA QUITAR RUIDO DE LA IMAGEN
        f_gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gra = cv2.morphologyEx(f_gray,cv2.MORPH_GRADIENT,kernel)
        gra = np.uint8((gra<5)*255)
        
        # f_gray = cv2.blur(f_gray,(3,3))
       
        # #ESTABLECEMOS LOS FRAMES EN PUNTO FLOTANTE PARA MAS PRECISIÓN 
        # f_float=np.float64(f_gray)
            
        # #CALCULAMOS LA TRANSFORMADA DISCRETA DE FOURIER EN 2D
        # transdormada_img=fft.fft2(f_float)
            
        # #desplazamos el centro de la frecuencia a los bordes, porque
        # #la transformada discreta de fourier lo hace. Es para que sea todo simetrico
        # #y las frecuencias coordinen
        # Fsh_Image=fft.fftshift(filt_gauss)
                
        # # Aplicamos el filtro gaussiano al espectro de la imagen (Imagen x Máscara)
        # fft_gauss=Fsh_Image*transdormada_img      
        # imagen_filtrada = fft.ifft2(fft_gauss)       
        # img_filter = cv2.normalize(abs(imagen_filtrada), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)        
        # img_umbral = cv2.adaptiveThreshold(img_filter,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,10)
        
        # #USAMOS LA MASCARA LOGICA PARA SEGMENTAR EL IC, 
        # #LUEGO CONVERVIMOS LOS PIXELES EN INT DE 8 BITS
        # mascara_2 = np.logical_and(frame_green_binary,img_umbral)
        # mascara_2 = np.uint8(mascara_2*255)
    
        # #OBTENCION DEL OBJETO DE INTERES
        #num_labels,labels,stats,centro = cv2.connectedComponentsWithStats(mascara_2,4,cv2.CV_32S)
        num_labels,labels,stats,centro = cv2.connectedComponentsWithStats(gra,4,cv2.CV_32S)
        mascara_3 = (np.argmax(stats[:,4][1:])+1==labels)
        mascara_3 = nd.binary_fill_holes(mascara_3).astype(int)
        mascara_3 = np.uint8(mascara_3*255)
        mascara_3 = cv2.dilate(mascara_3,kernel)
    
        # #CONTORNOS
        contorno,_ = cv2.findContours(mascara_3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contorno[0]
        
        #CONVEX HULL DEL IC
        hull = cv2.convexHull(cnt)
        puntos_convex = hull[:,0,:]
        alto,ancho = mascara_3.shape
        ar = np.zeros((alto,ancho))
        mascara_convex = np.uint8((cv2.fillConvexPoly(ar,puntos_convex,1))*255)
        
        seg = binview(frame_copy,mascara_convex,'m')
    
        x,y,w_ancho,h_alto = cv2.boundingRect(cnt)
        
        # #VENTANA QUE MUESTRA AL IC PROCESADO 
        cv2.namedWindow("IC")
        cv2.moveWindow("IC",50,50)
        cv2.rectangle(seg,(x-20,y-20),((x+w_ancho)+20,(y+h_alto)+20),color,grosor)
        cv2.imshow("IC",seg)
        
    if key & 0xFF == asci_w:
        cv2.destroyWindow("IC")
        print("VENTANA CERRADA")        
        
    cv2.imshow("imagen original",frame)

video.release()
cv2.destroyAllWindows()