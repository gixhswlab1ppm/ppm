import epd2in13d
import time
from PIL import Image,ImageDraw,ImageFont
import traceback
import threading
import collections

epd = epd2in13d.EPD()
epd.init()
epd.Clear(0xFF)
dev_lock = threading.Lock()

time_font = ImageFont.truetype('ArialNova.ttf', size=12)

# slot 0, 1, 2, 3, time, pause
field_locations = [[36,8,96,32], [36,36,96,60], [124,8,212,32], [124,36,212,60], [164,0,212,8], [8,64,96,88]]
field_font = ImageFont.truetype('ArialNova.ttf', size=24)
title_small = ImageFont.truetype('ArialNova.ttf', size=12)
last_img = Image.new('1', (epd2in13d.EPD_HEIGHT, epd2in13d.EPD_WIDTH), 255)

# def _update_triggerless():
#     time.sleep(2)
#     while True:
#         if ui_state == 3 or ui_state == 4:
#             update_display_partial(4, time.strftime('%H:%M:%S'), time_font)
#         time.sleep(1)
def render_welcome_screen():
    bg = Image.new('1', (epd2in13d.EPD_HEIGHT, epd2in13d.EPD_WIDTH), 255)
    dr = ImageDraw.Draw(bg)
    dr.text((36,36), 'Keep your paddle flat...', font=title_small, fill=0)
    with dev_lock:
        epd.display(epd.getbuffer(bg))
def render_profile_screen(user_id=0):
    bg = Image.new('1', (epd2in13d.EPD_HEIGHT, epd2in13d.EPD_WIDTH), 255)
    dr = ImageDraw.Draw(bg)
    dr.text((8,8), 'User Profiles', font=field_font, fill=0)
    for i in range(5):
        if i == user_id:
            dr.ellipse((8+i*(24+12), 36, 32+i*24, 36), outline='black', fill=0)
        else:
            dr.ellipse((8+i*(24+12), 36, 32+i*24, 36), outline='black')
    with dev_lock:
        epd.display(epd.getbuffer(bg))
def render_train_screen():
    bg = Image.new('1', (epd2in13d.EPD_HEIGHT, epd2in13d.EPD_WIDTH), 255)
    dr = ImageDraw.Draw(bg)
    dr.text((36,36), 'Keep swinging while', font=title_small, fill=0)
    dr.text((36,72), 'we learn from you...', font=title_small, fill=0)
    with dev_lock:
        epd.display(epd.getbuffer(bg))
def render_run_screen():
    bg = Image.new('1', (epd2in13d.EPD_HEIGHT, epd2in13d.EPD_WIDTH), 255)
    bg.paste(Image.open('paddle.jpg'), (8,8)) #24*24
    bg.paste(Image.open('wave.jpg'), (8,36)) #ball
    bg.paste(Image.open('swing.jpg'), (96,8)) #clock
    bg.paste(Image.open('thumb.jpg'), (96,36)) #round
    update_display_partial(5, "Playing")
    
def render_pause_screen():
    update_display_partial(5, "Paused")

def update_display_partial(field_id, field_val, field_font=field_font): # num, text
    # multi threads accessing update_field, will cause "roll-back" hazard if without lock!
    with dev_lock:
        global last_img
        background_img2 = last_img.copy()
        field_draw = ImageDraw.Draw(background_img2)
        field_draw.rectangle(tuple(field_locations[field_id]), fill=255)
        field_draw.text(tuple(field_locations[field_id][:2]), str(field_val), font=field_font, fill=0)
        cropped_img = background_img2.crop(field_locations[field_id])
        try:
            background_img2.paste(cropped_img, tuple(field_locations[field_id][:2]))
            epd.DisplayPartial(epd.getbuffer(background_img2))
            last_img = background_img2
        except:
            print('gg')
    print('updated field {0} with val {1}'.format(field_id, field_val))
# threading.Thread(target=_update_triggerless).start()



    
    # Image.resize()
    
    
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