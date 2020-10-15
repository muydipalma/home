---
layout: page
title: Apps con python 
---


Existen varias maneras de poner nuestros modelos en produccion, aca vamos a ver dos.

* [Flask](#flask)
* [Streamlit](#streamlit)


### Flask

la estructura mas basica de una app de flask:

```bash
from flask import Flask
app = Flask('mi app')
@app.route('/',methods=['GET','POST])
def main():
    print('hola!')
app.run(host='127.0.0.1',  port=5000)
```
Debemos poner todo esto en un archivo.py y ejecutarlo desde la consola:

```bash
python archivo.py
``` 

### Streamlit

Las apps de streamlit, son mas simples, es un script de python normal y vamos insertando los widgets que necesitamos en el cuerpo del script. Aca una lista de los widgets disponibles https://docs.streamlit.io/api.html
Un ejemplo basico:

```python
import streamlit as st
st.write('hola!')
```
Lo guardamos en un archivo.py y para ejecutarlo:

```bash
streamlit run archivo.py
``` 


* bokeh_flask.py: archivo con nuestro ejemplo de implementacion de un servidor de FLASK que recibe requests GET
* bokeh_st.py: archivo con la implementacion del ejemplo anterior hecho en streamlit
* stream_ej.ppy: archivo con la app de stream que implementa todo el ejercicio plantedo en la practica 1
* en la carpeta templates hay un index.html, necesario para el ejmplo de bokeh en flask de la practica 2




### Heroku 


Una vez que nuestras apps de streamlit funcionen como queremos, ya podemos poner el modelo en produccion.


1. Ir a [heroku.com](https://signup.heroku.com/) y registrarse.
2. Confirmar mail, crear contraseña y logearse.
3. Crear nueva app [aca](https://dashboard.heroku.com/new-app).
4. En el menu de la APP, ir a deploy method y seleccionar connect to hithub.
5. Asociar heroku con su cuenta de github, buscar el repositorio donde tenemos nuestra app y conectar.
6. Seleccionaremos automatic deploys para que se actualice automáticamente las versiones de nuestra app, y por ultimo le damos a DEPLOY BRANCH.


Para crear la maquia virtual en heroku, tenemos que especificar la version de python y de las librerias presentes en nuestra app, esto lo hacemos en los archivos requirements y runtime que estan en el repositorio de github que asociamos a nuestra APP de heroku.



Ejemplo del contenido de estos archivos esta en:

https://github.com/carabedo/geopami

```bash
app_st.py
requirements.txt
runtime.txt
Procfile
create_config.sh
``` 


Esta app esta funcionando en:

[geopami.herokuapp.com](https://geopami.herokuapp.com/)
