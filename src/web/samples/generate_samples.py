# src/web/samples/generate_samples.py
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

os.makedirs("src/web/samples", exist_ok=True)

# 1. Spliced Breaking News Card
img1 = Image.new("RGB", (640, 400), color=(26, 22, 18))
draw1 = ImageDraw.Draw(img1)
draw1.rectangle([0, 0, 640, 60], fill=(180, 40, 40))
draw1.text((30, 20), "BREAKING NEWS // LIVE PRESS MEET", fill=(255, 255, 255))
draw1.rectangle([20, 80, 620, 280], fill=(45, 38, 32))
draw1.text((40, 100), "OFFICIAL PRESS CONFERENCE", fill=(212, 185, 150))
draw1.rectangle([0, 340, 640, 400], fill=(15, 12, 10))
draw1.text((30, 360), "SCANDAL EXPOSED: LEADER ANNOUNCES SURRENDER", fill=(255, 255, 255))
img1.save("src/web/samples/spliced_news_card.jpg", "JPEG", quality=95)

# Load, alter lower banner with heavily compressed splice block
arr1 = cv2.imread("src/web/samples/spliced_news_card.jpg")
patch = np.zeros((60, 400, 3), dtype=np.uint8)
patch[:, :] = (20, 20, 200)
cv2.putText(patch, "FABRICATED SPLICED TEXT", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
_, enc = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 25])
dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
arr1[340:400, 100:500] = dec
cv2.imwrite("src/web/samples/spliced_news_card.jpg", arr1, [cv2.IMWRITE_JPEG_QUALITY, 95])

# 2. Recycled Historical Nature Photo (peaceful mountains used with volcanic disaster claim)
img2 = Image.new("RGB", (640, 420), color=(135, 206, 235))
draw2 = ImageDraw.Draw(img2)
draw2.polygon([(0, 300), (200, 120), (400, 300)], fill=(100, 110, 120))
draw2.polygon([(250, 300), (450, 150), (640, 300)], fill=(80, 90, 100))
draw2.rectangle([0, 300, 640, 420], fill=(40, 120, 180))
img2.save("src/web/samples/recycled_photo.jpg", "JPEG", quality=92)

# 3. Authentic Government Circular
img3 = Image.new("RGB", (600, 500), color=(250, 247, 242))
draw3 = ImageDraw.Draw(img3)
draw3.rectangle([20, 20, 580, 480], outline=(180, 160, 140), width=2)
draw3.text((150, 40), "GOVERNMENT OF TAMIL NADU", fill=(20, 20, 20))
draw3.text((170, 70), "HEALTH & FAMILY WELFARE DEPT", fill=(60, 60, 60))
draw3.line([(50, 100), (550, 100)], fill=(180, 160, 140), width=1)
draw3.text((50, 120), "OFFICIAL NOTIFICATION: DATED 12-09-2026", fill=(30, 30, 30))
draw3.text((50, 160), "Ref: Circular No. 402/HFW/2026", fill=(40, 40, 40))
draw3.text((50, 200), "Public Advisory: Routine Health Services Operating Normally.", fill=(20, 20, 20))
draw3.text((50, 230), "No changes in ration card or citizen health records.", fill=(40, 40, 40))
draw3.text((400, 420), "Sd/- Principal Secretary", fill=(20, 20, 20))
img3.save("src/web/samples/authentic_circular.jpg", "JPEG", quality=95)

print("Sample images generated successfully.")
