# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:16:50 2020

@author: TimAl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:15:27 2020

@author: TimAl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:33:46 2020

@author: TimAl
"""

import os
import sys

import multiprocessing
import time
import argparse

import numpy as np
import csv
import pandas as df
import requests
import json

from tqdm import tqdm
from PIL import Image
from io import BytesIO

import vrProjector
import utm

##### YOU CAN TECHNICALLY JUST IMPLEMENT THIS, ONLY THING YOU NEED TO LOOK OUT IS THE HARDCODED DIRECTORIES
##### OUTPUT AS WELL AS INPUT FOR THE WORKERS. ARE YOU GOING TO SPLIT UP YOUR 10k DIRECTORIES EVEN FURTHER?
##### WOULD BE GOOD TO RUN IT FROM BASE /IMAGES_NEW/DIR_001/ and output to /CUT_IMAGES/DIR_001/ to make sure
##### YOU DON'T CLUTTER EVERYTHING

#partly borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet

def stitcher_to_array(args, img):
    print('starting stitcher')
    
    #aangeven size van front back left right images
    #size = 240 geeft 240x240 images
    
    print('type image is:', type(img))
    
    
    #img_array = np.array(img)
    #img_array = img_array[200:, :]
    #img = Image.fromarray(np.uint8(img_array))
        
    #img.save('test2.jpg', 'JPEG')
    
    size = 224
    #print(1)
    eq = vrProjector.EquirectangularProjection()
    #print(2)
    eq.loadImage(img)
    #print(3)
    cb = vrProjector.CubemapProjection()
    #print(4)
    cb.initImages(size,size)

    cb.reprojectToThis(eq)
    
    
    #stitch je in 1 lange array
    stitched_array = np.concatenate((cb.front, cb.right, cb.back, cb.left, cb.front), axis=1)

    #cb_list = [cb.front, cb.right, cb.back, cb.left]
    print(stitched_array[0])
    print(len(stitched_array[0]))
    #wegknippen lucht en auto
    stitched_array = stitched_array[0:224, 340:1460]
    
    return stitched_array

def cutter(stitched_array, y, h, x, w):

    crop_img = stitched_array[y:y+h, x:x+w]
    
    return crop_img

def seq_24(args, image, base_name, output_dir, gps, postcode):
    print('starting seq with ')
    stitched_array = stitcher_to_array(args, image)
    
    
    ## Aanpassen aan je eigen preference
    
    ## Pitch is hoe hoog hou je je hoofd
    ## Yaw is horizontale rotatie
    #pitch = [1]
    #yaw = [1,2,3,4]
    yaw = [1,2,3,4]
    
    size = 224

    for j in yaw:
    
        
        x = (j - 1) * size
        w = size
        cropped_img = cutter(stitched_array, 0, size, x, w)             
        
        #cropped_img = cropped_img[:, :, [2, 1, 0]]
        
        
        name = base_name + '_yaw' + str(j)
        im = Image.fromarray(np.uint8(cropped_img))
        #cv.imwrite(name+'jpg', cropped_img)
        #print("121")
        directory = os.path.join(os.getcwd(),output_dir)
        if not os.path.exists(directory):
                    os.makedirs(directory)   
        print('saved ', name, '!')
        
        #saved de image naar de relevante directory
        filename = directory+ '/' + name+'.jpg'
        im.save(filename, 'JPEG')
        
        csv_filename = args.output_dir + '/' + name + '.jpg'
        
        #saved de gpscoordinaat naar de relevante csv
        with open('groundtruth_gps.csv', 'a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow([csv_filename, gps[0], gps[1], postcode])
                
                

class Consumer(multiprocessing.Process):

    def __init__(self, task_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            next_task()
            self.task_queue.task_done()
        return


def download(args, api_url, image_url, postcode, directory, full_dir):
    
    image_url = image_url[:-8] + '4000.jpg'
    #name = image_url[55:92]
    #print(image_url)
    try:
        ## Pull image from municipality API
        #print('Acessing API...')
        print('Currently handling ', image_url)
        image_response = requests.get(image_url)
        print(image_url, ' gave a ', image_response, ' image response')
        image = Image.open(BytesIO(image_response.content))
        #print(image_response)
        ## GPS from municipality API
        
        api_url = api_url[:69] + '_' + api_url[70:]
        
        api_response = requests.get(api_url)
        print(api_url, ' gave a ', api_response, ' api response')
        coordinates = json.loads(api_response.content)['geometry']['coordinates']
        id_of_pano = json.loads(api_response.content)['pano_id']
        print(id_of_pano)
        
        
        gps = [coordinates[1],coordinates[0]]
        #gps = utm.from_latlon(coordinates[1],coordinates[0])[:2]
        
        seq_24(args, image, id_of_pano, full_dir, gps, postcode)
                      
    
    except:
        print('sys info:',sys.exc_info())
        with open('errors.csv', 'a') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow([image_url, postcode])
        print(image_url)
        
class Task(object):
    def __init__(self, args, api_url, image_url, pc, directory, full_dir):
        #print('Task init')
        self.api_url = api_url
        self.image_url = image_url
        self.args = args
        self.dir = directory
        self.top_dir = full_dir
        self.postcode = pc
        
        
    def __call__(self):
        #print('Task call')
        download(self.args, self.api_url, self.image_url, self.postcode, self.dir, self.top_dir)
        
        #seq_24(self.args, self.info)
        #print('Inside call', self.info)
        
    def __str__(self):
        return self.info['filename']


def main(args):
    

    
    directory = os.path.join(os.getcwd(),args.top_dir)
    if not os.path.exists(directory):
                os.makedirs(directory)
                
    output_dir = args.output_dir
    
    full_directory = os.path.join(directory,output_dir)                
    if not os.path.exists(full_directory):
                os.makedirs(full_directory)                
    
    
    
      
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()

    # Start consumers
    num_consumers = multiprocessing.cpu_count() * 2
    
    print('Creating %d consumers' % num_consumers)
    consumers = [Consumer(tasks) for i in range(num_consumers)]
    for w in consumers:
        w.start()

    #### Give all the urls to the Taskforce
    data = df.read_csv(args.csv_filename)
    
    output_dir = args.output_dir
    a = int(output_dir[-2:])
    b = a-1
    
    amount = args.amount_of_panos
    
    data = data.loc[b*amount:(a*amount)-1]
    
    
    ### DEBUG
    #data = data.loc[:1]
    ###
    
    image_urls = data['url']
    pano_id = data['pano_id']
    postcodes = data['pc6']
    base_url = 'https://api.data.amsterdam.nl/panorama/panoramas/'
    urls = base_url + pano_id + '/'
    
    
    #### INSERTED FOR DEBUGGING PURPOSES
    #urls = urls[:2]
    #image_urls[:2]
    #### INSERTED FOR DEBUGGING PURPOSES
    
    
    print('Init zip..')
    print('About to process: ', len(urls), 'images')
    for api_url, image_url, postcode in zip(urls, image_urls, postcodes):
        
        tasks.put(Task(args, api_url, image_url, postcode, directory, full_directory))


    # Add a poison pill for each consumer
    for i in range(num_consumers):
        tasks.put(None)

    pbar = tqdm(total=tasks.qsize())

    last_queue = tasks.qsize()

    while tasks.qsize() > 0:
        diff = last_queue - tasks.qsize()
        pbar.update(diff)
        last_queue = tasks.qsize()
        time.sleep(0.2)


    # Wait for all of the tasks to finish
    tasks.join()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
     
    parser.add_argument('--top_dir', type=str, default='binnen_de_ring_dataset')
    
    
    ## output_dir should be in the form: dir001, dir002, dir003 etc..
    ## for cleanliness an output should pull no more than 10k panos, resulting
    ## in 240k perspective images. 
    
    parser.add_argument('--output_dir', type=str, default = 'dir_x')
    
    ## amount_of_panos specifies the amount of panoramas pulled by this instance
    ## of the script
    
    parser.add_argument('--amount_of_panos', type=int, default = 6000)
    
    ## name of the csvfile the urls are read from
    
    parser.add_argument('--csv_filename', type=str, default = 'groundtruths.csv')
    
    args = parser.parse_args()
    
    main(args)
