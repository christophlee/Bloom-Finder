import streamlit as st
import numpy as np
import pandas as pd
import datetime

from shapely.geometry import Point, Polygon
import geopandas as gpd
import geopy
import pickle

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut
from geopy.distance import distance
from geopy import Point

import pydeck as pdk

import matplotlib as mpl 
import matplotlib.pyplot as plt 
from matplotlib import cm 

#import pyproj
#import rasterio

import re

from daylength import daylength

class raster_dataset:
    
    def __init__(self, filename):#, window=None):
        #self.window = window
        self.src = rasterio.open(filename)
        self.src_coord = pyproj.Proj(self.src.crs)
        self.lonlat = pyproj.Proj(init='epsg:4326')
        self.data = self.src.read(1)#, window=self.window)

    def get_gps (self, row, col):
        east, north = self.src.xy(row,col) # image --> spatial coordinates
        lon,lat = pyproj.transform(self.src_coord, self.lonlat, east, north)
        value = self.data[row, col]
        return lon, lat
    
    # input: longitude, latitude (gps coordinate)
    # return: data value at input location(s)
    def get_value (self, lon, lat):
        east,north = pyproj.transform(self.lonlat, self.src_coord, lon, lat)
    
        # What is the corresponding row and column in our image?
        row, col = self.src.index(east, north) # spatial --> image coordinates
        #print(f'row,col=\t\t({row},{col})')
    
        # What is the value at that index?
        value = self.data[row, col]
        return value
    
    def get_rc (self, lon, lat):
        east,north = pyproj.transform(self.lonlat, self.src_coord, lon, lat)
        return self.src.index(east, north)
    
    def write_file (self, filename, data=None):
        
        if data is None:
            data = self.data
    
        with rasterio.Env():
        
            # Write an array as a raster band to a new 8-bit file. For
            # the new file's profile, we start with the profile of the source
            profile = self.src.profile
        
            # And then change the band count to 1, set the
            # dtype to uint8, and specify LZW compression.
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw')
        
            with rasterio.open(filename, 'w', **profile) as dst:
                dst.write(data.astype(rasterio.float32), 1)


#st.markdown(f"<style> div { background-image: url('Walker_canyon_wildflowers.jpg'); } </style>",unsafe_allow_html=True) 
#st.markdown(f'<div style="background-image: url(\'Walker_canyon_wildflowers.jpg\');">',unsafe_allow_html=True)

# App title and description
app_title = 'Bloom Finder'

st.title(app_title)

st.write(app_title+' predicts wildflower blooms in popular wilderness areas in California.  To get started, select your location and desired visit date.')

# Date selector.  Ensure date selected is acceptable.
valid_date = False

today = datetime.date.today()
visit_date = st.date_input('Select visit date:')
max_date = today + datetime.timedelta(days=300)
min_date = datetime.date(2009,1,1)
if visit_date > max_date:
    st.error('Error: Date must be before %s (30 days from today\'s date)' % max_date)
elif visit_date < min_date:
    st.error('Error: date cannot be earlier than 2009-01-01')
    valid_date = False
else:
    st.success('Date Selected: %s' % visit_date)
    valid_date = True

# Location selector
city = st.text_input("Select your location in California:", "San Francisco")

# parse location to valid name and lookup GPS location
city = ' '.join(city.split()).title()

geolocator = Nominatim(user_agent='christoph@appliedanalytics.xyz')

location = None
valid_location = False
# make sure parsed place name contains only alphabetic charaters (or hyphen, apostrophe, space)
valid_ascii = bool(re.match("^[A-Za-z-' ]*$", city))

# load the parks dataframe
parks = pd.read_pickle('parks_full.pkl')

# load the model from disk
logmodel = pickle.load(open('logmodel_0608.sav', 'rb'))

# do geolocation only if location string is valid
if (valid_ascii):

    try:
        # restrict search to California
        location = geolocator.geocode(city + ", California, USA")
    except GeocoderTimedOut as e:
        st.error('Error: geocoder timed out, please try again.')
        valid_location = False

print_city = ''

if (location == None):
    st.error('Error: %s, California could not be found. Try a different city.' % city)
    valid_location = False
else:
    print_city = city
    if len(city):
        print_city += ', '
    st.success(('Location: %s'+'California (%lf,%lf)')%(print_city,location.latitude,location.longitude))
    valid_location = True


search_radius = st.number_input("How many miles are you willing to travel from your location?", min_value=0, value=100)
valid_distance = search_radius>=0

#print_city = city
#if len(city):
#    print_city += ', '

if (valid_location):
    if (valid_distance):
        search_radius = float(search_radius)    
        st.success("Searching within %.0lf miles of %sCalifornia"%(search_radius,print_city))
    else:
        st.error('Error: enter a valid number of miles.')    

if (valid_date and valid_location and valid_distance):
    lat = location.latitude
    lon = location.longitude

    st.write("Awesome, here is our wildflower bloom prediction for parks near %s on %s!"%(print_city+ "California", visit_date))

    #st.write('%lf'%lat)
    #st.write('%lf'%lon)

    # display list of parks
    lp = Point((lat,lon))

    #parks['Distance'] = parks[['Latitude','Longitude']].apply(lambda x: distance(Point((x.Latitude,x.Longitude)),lp).kilometers, axis=1)
    #parks['Day_of_Year'] = visit_date.timetuple().tm_yday

    #parks['daylength'] = parks.apply(lambda x: daylength(x.Day_of_Year,x.Latitude),axis=1)

    ## lookup agdd
    #parks['AGDD0_100'] = np.nan
    #agdd_path = 'prism/daily/agdd0/%s/PRISM_agdd0_%s%02d%02d.tif'
    #
    ## read in aggd0 file
    #agdd_data = raster_dataset(agdd_path % (visit_date.year,visit_date.year,visit_date.month,visit_date.day))
    #parks['AGDD0_100'] = parks.apply(lambda x: agdd_data.get_value(x.Longitude,x.Latitude), axis=1)

    ## lookup cum_ppt_100
    #parks['cum_ppt_100'] = np.nan
    #cum_ppt_100_path = 'prism/daily/cum_ppt_100/%s/PRISM_cum_ppt_100_%s%02d%02d.tif'
    #
    ## read in aggd0 file
    #cum_ppt_100_data = raster_dataset(cum_ppt_100_path % (visit_date.year,visit_date.year,visit_date.month,visit_date.day))
    #parks['cum_ppt_100'] = parks.apply(lambda x: cum_ppt_100_data.get_value(x.Longitude,x.Latitude), axis=1)

    parks_visit_date = parks[parks.Day_of_Year==visit_date.timetuple().tm_yday].copy(deep=True)
    parks_visit_date['Distance (miles)'] = parks_visit_date[['Latitude','Longitude']].apply(lambda x: distance(Point((x.Latitude,x.Longitude)),lp).miles, axis=1)

    parks_visit_date = parks_visit_date[parks_visit_date['Distance (miles)']<search_radius]
    parks_visit_date.reset_index(inplace=True)

    predictions = logmodel.predict(parks_visit_date[['elevation_srtm','daylength','AGDD0_100','cum_ppt_100']])
    probs = logmodel.predict_proba(parks_visit_date[['elevation_srtm','daylength','AGDD0_100','cum_ppt_100']])
    probs = pd.DataFrame(probs,columns=['probability_%d'%n for n in range(2)])

    # reform test set to include predictions from model
    #parks['labels'] = y_test
    parks_visit_date['predictions'] = predictions
    for i,c in enumerate(probs.columns):
        parks_visit_date[c] = probs.values[:,i]

    parks_visit_date['Bloom Score'] = (parks_visit_date.probability_1/0.52*100.).round(2)

    df_format = {'Distance (miles)':"{:.2f}", 'Bloom Score':"{:.2f}"}

    st.write(parks_visit_date[['Name','Distance (miles)','Bloom Score']].sort_values(by='Bloom Score',ascending=False).reset_index(drop=True).style.format(df_format))

    cols = parks_visit_date.columns.values
    for i,c in enumerate(cols):
        if c=='Latitude':
            cols[i] = 'lat'
        if c=='Longitude':
            cols[i] = 'lon'

    parks_visit_date.columns = cols
    parks_visit_date['subtext'] = '<br />Bloom Score: '+parks_visit_date['Bloom Score'].astype(str)

    alpha = 160
    ncolors = 50            

    cmap=plt.get_cmap('brg')
    norm = mpl.colors.Normalize(vmin=0, vmax=100)    
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    def get_rgb_string(val):
        r,g,b,a = scalarMap.to_rgba(val)
        return '[%lf,%lf,%lf,%d]'%(r*255,g*255,b*255,alpha)

    layers = []

    cats,bins = pd.cut(parks_visit_date['Bloom Score'],np.linspace(0,100,ncolors+1),retbins=True)
    parks_visit_date['cats'] = cats
    count = 0
    for key,group in parks_visit_date.groupby('cats'):
        layers.append(pdk.Layer(
            'ScatterplotLayer',
            data=group[['lat','lon','Bloom Score','Name','subtext']],
            pickable=True,
            get_position='[lon, lat]',
            get_color=get_rgb_string(bins[count]),
            get_radius=5000,
            radiusMinPixels=5,
            radiusMaxPixels=10,
            ))
        count += 1

    #"url": "https://img.icons8.com/plasticine/100/000000/marker.png",
    icon_data = {
        "url": "https://img.icons8.com/android/96/000000/marker.png",
        "width": 128,
        "height":128,
        "anchorY": 128
    }

    icon_df = pd.DataFrame.from_dict({'lat':[location.latitude],'lon':[location.longitude],'Name':[print_city+'California'],'Bloom Score':[0.0],'subtext':''})
    icon_df['icon_data'] = None
    icon_df.icon_data[0] = icon_data

    layers.append(pdk.Layer(
        'IconLayer',
        data = icon_df,
        get_icon='icon_data',
        get_size=3,
        pickable=True,
        size_scale=15,
        get_position='[lon, lat]'))

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
         initial_view_state=pdk.ViewState(
             latitude=location.latitude,
             longitude=location.longitude,
             zoom=8,
             pitch=0,
             bearing=0
         ),
         layers=layers,
         #tooltip={"html": "<b>{Name}</b><br />Bloom Score: {Bloom Score}", "style": {"color": "white"}}
         tooltip={"html": "<b>{Name}</b>{subtext}", "style": {"color": "white"}}
         ))

#st.markdown(f'</div>',unsafe_allow_html=True)
