import azure.functions as func
import logging
import requests
import os
from io import BytesIO
from PIL import Image, ImageDraw
import base64

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

def process_image(image_data, ENDPOINT, PREDICTION_KEY, PROJECT_ID, ITERATION_NAME):
    url = f"{ENDPOINT}/customvision/v3.0/Prediction/{PROJECT_ID}/detect/iterations/{ITERATION_NAME}/image"
    headers = {
        'Content-Type': 'application/octet-stream',
        'Prediction-Key': PREDICTION_KEY
    }
    response = requests.post(url, headers=headers, data=image_data)
    response.raise_for_status()
    return response.json()

def draw_bounding_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    for pred in predictions['predictions']:
        if pred['probability'] > 0.5:  # Limiar de confiança
            left = pred['boundingBox']['left'] * image.width
            top = pred['boundingBox']['top'] * image.height
            width = pred['boundingBox']['width'] * image.width
            height = pred['boundingBox']['height'] * image.height
            draw.rectangle([left, top, left + width, top + height], outline='red', width=2)
            draw.text((left, top), f"{pred['tagName']} ({pred['probability']:.2f})", fill='red')
    return image

@app.route(route="fruteirafuncion4")

def fruteirafuncion4(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        # Obter o corpo JSON
        req_body = req.get_json()
        # Obter a string Base64 da imagem
        base64_string = req_body.get("base64_string") 

        # Configurações do Azure Custom Vision
        ENDPOINT = req_body.get("ENDPOINT") 
        PREDICTION_KEY = req_body.get("PREDICTION_KEY") 
        PROJECT_ID = req_body.get("PROJECT_ID") 
        ITERATION_NAME = req_body.get("ITERATION_NAME") 


        # Converter Base64 para bytes
        image_data = base64.b64decode(base64_string)

        image = Image.open(BytesIO(image_data))

        predictions = process_image(image_data, ENDPOINT, PREDICTION_KEY, PROJECT_ID, ITERATION_NAME)
        processed_image = draw_bounding_boxes(image, predictions)

        img_byte_arr = BytesIO()
        processed_image.save(img_byte_arr, format='JPEG')
        # Converter o buffer de bytes para Base64 para suportar pelo PowerApps
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        # Retornar a imagem como Base64
        return func.HttpResponse(img_base64, mimetype="text/plain") 

    except Exception as e:
        logging.error(f"Erro ao processar a imagem: {e}")
        return func.HttpResponse(f"Erro: {e}", status_code=500)