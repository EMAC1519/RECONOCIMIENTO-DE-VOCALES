from PIL import Image
import sys
import os

# Caracteres de oscuro a claro (puedes modificarlos)
ASCII_CHARS = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

def resize_image(image, new_width=100):
    """Redimensiona la imagen manteniendo la proporción"""
    width, height = image.size
    ratio = height / width
    new_height = int(new_width * ratio * 0.55)  # 0.55 corrige la proporción de los caracteres
    return image.resize((new_width, new_height))

def grayify(image):
    """Convierte la imagen a escala de grises"""
    return image.convert("L")

def pixels_to_ascii(image):
    """Convierte cada pixel en un carácter ASCII según su brillo"""
    pixels = image.getdata()
    chars = "".join([ASCII_CHARS[pixel * (len(ASCII_CHARS) - 1) // 255] for pixel in pixels])
    return chars

def image_to_ascii(path, width=100):
    """Función principal: carga la imagen y la convierte a ASCII"""
    try:
        image = Image.open(path)
    except Exception as e:
        print(f"Error al abrir la imagen: {e}")
        return
    
    image = resize_image(image, width)
    image = grayify(image)
    
    ascii_str = pixels_to_ascii(image)
    
    # Dividir la cadena en líneas según el ancho
    img_width = image.width
    ascii_img = "\n".join(
        ascii_str[i:i + img_width] 
        for i in range(0, len(ascii_str), img_width)
    )
    
    return ascii_img

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python ascii_art.py <ruta_de_la_imagen> [ancho]")
        print("Ejemplo: python ascii_art.py foto.jpg 120")
        sys.exit(1)
    
    path = sys.argv[1]
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    if not os.path.exists(path):
        print(f"No se encontró el archivo: {path}")
        sys.exit(1)
    
    resultado = image_to_ascii(path, width)
    if resultado:
        print(resultado)