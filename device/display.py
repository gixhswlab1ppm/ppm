import epd2in13d
import time
from PIL import Image,ImageDraw,ImageFont
import traceback
import threading
import collections

font15 = ImageFont.truetype('ArialNova.ttf',size=20)

epd = epd2in13d.EPD()
epd.init()
epd.Clear(0xFF)

frame_buf = collections.deque(maxlen=4)
lock = threading.Lock()

def draw(img, dim=(0,0,0,0), topleft=(0,0)):
    with lock:
        frame_buf.appendleft((img, dim, topleft)) # dim: tuple of 4

def render_daemon():
    while True:
        if len(frame_buf) > 0:
            with lock:
                frame_img, frame_dim, frame_topleft = frame_buf.pop()
            img = Image.new('1', (epd2in13d.EPD_HEIGHT, epd2in13d.EPD_WIDTH), 255)
            draw = ImageDraw.Draw(img)
            draw.rectangle(frame_dim, fill=255)
            cropped_img = frame_img.crop(list(frame_dim))
            cropped_img = 
            epd.DisplayPartial(epd.getbuffer)



try:
    epd.Clear(0xFF)
    time_image = Image.new('1', (epd2in13d.EPD_HEIGHT, epd2in13d.EPD_WIDTH), 255)
    time_draw = ImageDraw.Draw(time_image)
    while (True):
        time_draw.rectangle((10, 10, 120, 50), fill = 255) # the "size" of the box
        time_draw.text((10, 10), "Swing count", font = font15, fill = 0)
        time_draw.text((10, 30), time.strftime('%H:%M:%S'), font = font15, fill = 0)
        newimage = time_image.crop([10, 10, 120, 50]) #  left, upper, right, and lower pixel # 212*104
        newimage = time_image.paste(newimage, (10,10))  
        epd.DisplayPartial(epd.getbuffer(time_image))
        
    epd.sleep()
except Exception as e:
    print('traceback.format_exc():\n%s',traceback.format_exc())
    exit()

    
    
    # # Drawing on the image
    # image = Image.new('1', (epd2in13d.EPD_HEIGHT, epd2in13d.EPD_WIDTH), 255)  # 255: clear the frame
    
    # draw = ImageDraw.Draw(image)    
    # # draw.rectangle([(0,0),(50,50)],outline = 0)
    # # draw.rectangle([(55,0),(100,50)],fill = 0)
    # # draw.line([(0,0),(50,50)], fill = 0,width = 1)
    # # draw.line([(0,50),(50,0)], fill = 0,width = 1)
    # # draw.chord((10, 60, 50, 100), 0, 360, fill = 0)
    # # draw.ellipse((55, 60, 95, 100), outline = 0)
    # # draw.pieslice((55, 60, 95, 100), 90, 180, outline = 0)
    # # draw.pieslice((55, 60, 95, 100), 270, 360, fill = 0)
    # # draw.polygon([(110,0),(110,50),(150,25)],outline = 0)
    # # draw.polygon([(190,0),(190,50),(150,25)],fill = 0)
    # draw.text((110, 60), 'e-Paper demo', font = font15, fill = 0)
    # draw.text((110, 80), 'Hello world', font = font15, fill = 0)
    # epd.display(epd.getbuffer(image))
    # time.sleep(2)
    
    # # read bmp file 
    # # epd.Clear(0xFF)
    # image = Image.open('2in13d.bmp')
    # epd.display(epd.getbuffer(image))
    # time.sleep(2)
    
    # # read bmp file on window
    # epd.Clear(0xFF)
    # image1 = Image.new('1', (epd2in13d.EPD_WIDTH, epd2in13d.EPD_HEIGHT), 255)  # 255: clear the frame
    # bmp = Image.open('100x100.bmp')
    # image1.paste(bmp, (0,10))    
    # epd.display(epd.getbuffer(image1))
    # time.sleep(2)