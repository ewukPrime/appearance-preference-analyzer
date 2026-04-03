import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pathlib import Path
from loguru import logger
import cv2
from groundingdino.util.inference import load_model, load_image, predict
from groundingdino.util.box_ops import box_cxcywh_to_xyxy
import supervision as sv
from torchvision.ops import box_convert

class ImagePreprocessor:
    def __init__(self):
        data_dir = Path('data/logs')
        logger.add(data_dir / 'ImagePreprocessor.log', rotation='1 MB', level='DEBUG', encoding='utf-8')
        logger.info('Сессия начата')



    def without_background(self, img_path):
        logger.info('Начало обрезания фона')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam_cfg = "C:/Dev/MainProject2/.venv/Lib/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        sam_weights = "models/sam2.1_l.pt"
        predictor = SAM2ImagePredictor(build_sam2(sam_cfg, sam_weights, device=device))

        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        predictor.set_image(image)
        h, w, _= image.shape

        dino_model = load_model("C:/Dev/MainProject2/.venv/Lib/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py", "models/groundingdino_swint_ogc.pth")
        _, image_tensor = load_image(img_path)
        text_promt = "girl ."
        boxes, logits, phrases = predict(
            model=dino_model,
            image=image_tensor,
            caption=text_promt,
            box_threshold=0.1,
            text_threshold=0.1,
            device=device
        )
        if boxes.numel() > 0:
            boxes = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
            boxes = boxes.cpu().numpy() * np.array([w, h, w, h])
        else: logger.error("Объекта не найдено!")

        masks, scores, _ = predictor.predict(
            box=boxes[0:1], 
            multimask_output=False
        )
        mask = masks.squeeze()




        # input_point = np.array([[w/2, h/2]])
        # input_label = np.array([1])
        # masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label)
        # mask = masks[np.argmax(scores)]



        bg_color = np.array([128, 128, 128])
        image = image.astype(float)
        
        final_img = (image * np.expand_dims(mask, axis=2) + bg_color * (1 - np.expand_dims(mask, axis=2))).astype(np.uint8)

        result = Image.fromarray(final_img)

        logger.success('Фон обрезан')
        result.save("data/output_for_clip.jpg")





    def without_background2(self, img_path):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Путь к конфигу и весам (используем ваши версии 2.1)
        model_cfg = "C:/Dev/MainProject2/.venv/Lib/site-packages/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" 
        sam2_checkpoint = "models/sam2.1_l.pt"

        predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device=device))

        # Загружаем картинку
        image = Image.open(img_path).convert("RGB")
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        predictor.set_image(img_array)
        
        # 1. Задаем рамку (охватывает всё тело)
        input_box = np.array([5, 5, w - 5, h - 5]) 

        # 2. Добавляем "контрольные точки" (Point Prompting)
        # Ставим точки там, где чаще всего возникают "вырезы": 
        # Центр лица, центр волос, центр туловища.
        input_points = np.array([
            [w // 2, h // 3],  # Примерно область головы/лица
            [w // 2, h // 2],  # Центр изображения
            [w // 2, h * 3 // 4] # Область одежды
        ])
        # Все точки помечаем как "1" (это точно объект)
        input_labels = np.array([1, 1, 1])

        # 3. Передаем ВСЁ вместе
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_box, # Рамка + Точки работают вместе идеально
            multimask_output=False 
        )


        if isinstance(masks, torch.Tensor):
            mask_uint8 = (masks.cpu().numpy().squeeze() * 255).astype(np.uint8)
        else:
            mask_uint8 = (masks.squeeze() * 255).astype(np.uint8)

        # 2. Убираем "шум" (мелкие точки вокруг волос)
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel) # убирает мелкие точки
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel) # заделывает мелкие дырки внутри

        # 3. Сглаживание краев (Soft Edge) — ОЧЕНЬ важно для CLIP
        # Это сделает переход от волос к фону плавным
        blur_size = 5 # подберите 3, 5 или 7
        mask_blurred = cv2.GaussianBlur(mask_cleaned, (blur_size, blur_size), 0) / 255.0

        # 4. Финальное смешивание (Альфа-канал вручную)
        img_float = img_array.astype(float)
        bg_color = np.array([128, 128, 128], dtype=float) # Серый фон

        # Применяем мягкую маску для смешивания пикселей на границах
        # Формула: результат = объект * маска + фон * (1 - маска)
        for i in range(3): # по цветам RGB
            img_float[:, :, i] = img_float[:, :, i] * mask_blurred + bg_color[i] * (1 - mask_blurred)

        final_img = img_float.astype(np.uint8)
        result = Image.fromarray(final_img)
        result.save("data/output_for_clip.jpg")



    # def without_background3(self):
    #     bgr_image = self.bgr_image
    #     image = self.mp_image
    #     base_options = mpp.BaseOptions(model_asset_path='models/deeplab_v3.tflite')
    #     options = mpv.ImageSegmenterOptions(
    #         base_options=base_options,
    #         running_mode=mpv.RunningMode.IMAGE,
    #         output_category_mask=True,
    #         output_confidence_masks=True
    #     )

    #     with mpv.ImageSegmenter.create_from_options(options) as segmenter:
    #         result = segmenter.segment(image)

    #     mask = result.category_mask.numpy_view()
    #     cv2.imshow('Selfie Mask', cv2.bitwise_and(bgr_image, bgr_image, mask=mask))
    #     cv2.waitKey(0)



    # def without_background2(self):
    #     # session = new_session('isnet-general-use')
    #     result = remove(self.bgr_image)

    #     cv2.imshow('Rembg Result', result)
    #     cv2.waitKey(0)
