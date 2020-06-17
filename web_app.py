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

from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier

import pydeck as pdk

import matplotlib as mpl 
import matplotlib.pyplot as plt 
from matplotlib import cm 

import re
from daylength import daylength

@st.cache(allow_output_mutation=True)
def load_data(model = 'rf'):

    model = pickle.load(open('rfmodel_0616.sav','rb'))
    #model = pickle.load(open('logmodel_0616.sav','rb'))

    x_mean = np.load('x_mean_0616.npy')
    x_std = np.load('x_std_0616.npy')

    return model, x_mean, x_std

model, x_mean, x_std = load_data()

parks = pd.read_pickle('data/parks_full.pkl')

interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
scale = lambda x: (x-x_mean)/x_std
transform_input = lambda x: scale(interaction.fit_transform(x))

# load the parks dataframe
#parks = pd.read_pickle('data/parks_full.pkl')

# load the model from disk
#logmodel = pickle.load(open('logmodel_0608.sav', 'rb'))

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

    # display list of parks
    lp = Point((lat,lon))

    parks_visit_date = parks[parks.Day_of_Year==visit_date.timetuple().tm_yday].copy(deep=True)
    parks_visit_date['Distance (miles)'] = parks_visit_date[['Latitude','Longitude']].apply(lambda x: distance(Point((x.Latitude,x.Longitude)),lp).miles, axis=1)

    parks_visit_date = parks_visit_date[parks_visit_date['Distance (miles)']<search_radius]
    parks_visit_date.reset_index(inplace=True)

    predictions = model.predict(transform_input(parks_visit_date[['elevation_srtm','daylength','AGDD0_100','cum_ppt_100']]))
    probs = model.predict_proba(transform_input(parks_visit_date[['elevation_srtm','daylength','AGDD0_100','cum_ppt_100']]))
    probs = pd.DataFrame(probs,columns=['probability_%d'%n for n in range(2)])

    # reform test set to include predictions from model
    #parks['labels'] = y_test
    parks_visit_date['predictions'] = predictions
    for i,c in enumerate(probs.columns):
        parks_visit_date[c] = probs.values[:,i]

    parks_visit_date['Bloom Score'] = (parks_visit_date.probability_1/.5*100.).round(2)

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
