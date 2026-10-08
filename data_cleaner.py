import pandas as pd
import io
import zipfile
import requests
import os
import re

# ================================================================
# Estaciones meteorologicas
# ================================================================

def limpiar_estaciones():
  df_estaciones = pd.read_fwf(
    "./datasets/estaciones_smn.txt",
    encoding="latin-1"
  )

  df_estaciones = df_estaciones.iloc[1:].copy()

  df_estaciones.columns = [
    "nombre",
    "provincia",
    "latitud",
    "longitud",
    "altura",
    "nro",
    "nro_oaci"
  ]

  # Arregla nombres que se cortaron
  df_estaciones = df_estaciones[~df_estaciones["nombre"].isin(["ERO", "RO"])].copy()

  df_estaciones["nombre"] = df_estaciones["nombre"].replace({
    "PRESIDENCIA ROQUE SAENZ PEÑA A": "PRESIDENCIA ROQUE SAENZ PEÑA AERO",
    "ESCUELA DE AVIACION MILITAR AE": "ESCUELA DE AVIACION MILITAR AERO"
  })

  # Transformar columnas de latitud y longitud

  def convertir_coordenada(value):
    partes = str(value).split()

    grados = float(partes[0])
    minutos = float(partes[1])

    signo = -1 if grados < 0 else 1

    return signo * (abs(grados) + minutos / 60)

  df_estaciones["latitud"] = df_estaciones["latitud"].apply(convertir_coordenada)
  df_estaciones["longitud"] = df_estaciones["longitud"].apply(convertir_coordenada)

  df_estaciones.to_csv(
    "./datasets/estaciones_meteorologicas.csv",
    index=False
  )
  
  print('Dataset de estaciones procesado correctamente.')

# ================================================================
# Temperaturas 365 dias
# ================================================================

def descargar_temps_365d(directorio_destino: str = "./datasets") -> str:
  """
  Descarga el dataset de temperaturas de los últimos 365 días del SMN,
  extrae el archivo .txt contenido en el ZIP y lo guarda dentro del directorio especificado.

  Args:
    directorio_destino (str): Carpeta donde se guardará el archivo. Por defecto './datasets'.

  Returns:
    str: Ruta completa del archivo guardado si fue exitoso, o None si falló.
  """
  url = "https://ssl.smn.gob.ar/dpd/zipopendata.php?dato=regtemp"
  
  # Asegurar que el directorio de destino exista
  os.makedirs(directorio_destino, exist_ok=True)
  ruta_salida = os.path.join(directorio_destino, "temperaturas_365d.txt")
  
  try:
    print(f"=" * 60)
    print("Descargando temperaturas de 365 días desde el SMN...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
      archivos = z.namelist()
      archivo_txt = next((f for f in archivos if f.endswith(".txt")), None)

      if archivo_txt:
        contenido = z.read(archivo_txt)
        with open(ruta_salida, "wb") as f:
          f.write(contenido)
        
        print(f"✅ Archivo extraído y guardado correctamente en '{ruta_salida}'")
        print(f"=" * 60)
        return ruta_salida
      else:
        print("❌ No se encontró ningún archivo .txt dentro del ZIP descargado.")
        print(f"=" * 60)
        return None

  except Exception as e:
    print(f"❌ Error al descargar o extraer el archivo: {e}")
    print(f"=" * 60)
    return None

def limpiar_temps_365d():
  df_temps = pd.read_fwf(
  "./datasets/temperaturas_365d.txt",
  encoding="latin-1",
  skiprows=3,
  names=["fecha", "tmax", "tmin", "estacion"],
  dtype={"fecha":str}
)
  
  # Transformar las fechas a formato YYYY-MM-DD

  df_temps["fecha"] = pd.to_datetime(
    df_temps["fecha"].str.strip(),
    format="%d%m%Y",
    errors="coerce"
  ).dt.date

  df_temps.to_csv(
    "./datasets/temperaturas_365d.csv",
    index=False
  )
  
  print('Dataset de temperaturas procesado correctamente.')

# ================================================================
# Estadisticas normales 1991 - 2020
# ================================================================

def limpiar_normales():
  df = pd.read_excel(
    "./datasets/estadisticas_normales_1991-2020.xlsx",
    skiprows=4,
    names=["estacion", "valor", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre", ],
  )

  # Limpiar strings
  df['estacion'] = df['estacion'].str.replace('*', '')
  df['valor'] = df['valor'].str.replace(' (2011-2020)', '')

  # Transformar 'S/D'
  # Hay bastantes filas sin registros
  df = df.replace("S/D", pd.NA)

  # df[df['enero'].isnull()].head(20)

  df.to_csv(
    "./datasets/estadisticas_normales_1991-2020.csv",
    index=False
  )
  
  print('Dataset de normales 1991-2020 procesado correctamente.')

# ================================================================
# Pronostico a 5 dias
# ================================================================

def descargar_pronosticos_5d(directorio_destino: str = "./datasets") -> str:
  """
  Descarga el dataset de pronósticos a 5 días del SMN,
  extrae el archivo .txt contenido en el ZIP y lo guarda dentro del directorio especificado.

  Args:
    directorio_destino (str): Carpeta donde se guardará el archivo. Por defecto './datasets'.

  Returns:
    str: Ruta completa del archivo guardado si fue exitoso, o None si falló.
  """
  url = "https://ssl.smn.gob.ar/dpd/zipopendata.php?dato=pron5d"
  
  # Asegurar que el directorio de destino exista
  os.makedirs(directorio_destino, exist_ok=True)
  ruta_salida = os.path.join(directorio_destino, "pronosticos_5d.txt")
  
  try:
    print(f"=" * 60)
    print("Descargando pronósticos a 5 días desde el SMN...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
      archivos = z.namelist()
      archivo_txt = next((f for f in archivos if f.endswith(".txt")), None)

      if archivo_txt:
        contenido = z.read(archivo_txt)
        with open(ruta_salida, "wb") as f:
          f.write(contenido)
        
        print(f"✅ Archivo extraído y guardado correctamente en '{ruta_salida}'")
        print(f"=" * 60)
        
        return ruta_salida
      else:
        print("❌ No se encontró ningún archivo .txt dentro del ZIP descargado.")
        print(f"=" * 60)
        return None

  except Exception as e:
    print(f"❌ Error al descargar o extraer el archivo: {e}")
    return None

def limpiar_pronosticos_5d(
  ruta_txt: str = "./datasets/pronosticos_5d.txt",
  ruta_csv: str = "./datasets/pronosticos_5d.csv"
) -> pd.DataFrame:
  """
  Lee el archivo .txt de pronóstico a 5 días del SMN, lo limpia y exporta a .csv
  con las columnas: [estacion, fecha, hora, temperatura, viento_dir, viento_vel, precipitacion].

  Args:
    ruta_txt (str): Ruta del archivo .txt de origen.
    ruta_csv (str): Ruta donde se guardará el archivo .csv generado.

  Returns:
    pd.DataFrame: DataFrame limpio con los datos procesados.
  """
  # Mapeo de meses
  MESES = {
    'ENE': '01', 'FEB': '02', 'MAR': '03', 'ABR': '04',
    'MAY': '05', 'JUN': '06', 'JUL': '07', 'AGO': '08',
    'SEP': '09', 'OCT': '10', 'NOV': '11', 'DIC': '12'
  }

  # Regex para capturar las filas de datos:
  # Ej: "01/OCT/2026 00Hs. 10.5 172 | 14 0.1"
  pattern = re.compile(
    r'^\s*(\d{2})/([A-Z]{3})/(\d{4})\s+(\d{2})Hs\.\s+([-+]?\d+(?:\.\d+)?)\s+(\d+)\s*\|\s*(\d+)\s+([-+]?\d+(?:\.\d+)?)\s*$'
  )

  registros = []
  estacion_actual = None
  posible_estacion = None

  if not os.path.exists(ruta_txt):
    print(f"❌ Error: El archivo '{ruta_txt}' no existe.")
    return None

  with open(ruta_txt, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
      line_str = line.strip()
      if not line_str:
        continue

      # Si encontramos la línea divisoria '='
      if line_str.startswith("==="):
        if posible_estacion:
          estacion_actual = posible_estacion
          posible_estacion = None
        continue

      # Evaluar si es una línea de datos con el patrón regex
      match = pattern.match(line)
      if match:
        dia, mes_str, anio, hora, temp, v_dir, v_vel, precip = match.groups()

        # 1. Formatear Fecha: DD/MMM/YYYY -> YYYY-MM-DD
        mes_num = MESES.get(mes_str.upper(), '01')
        fecha_formatted = f"{anio}-{mes_num}-{dia}"

        # 2. Formatear Hora: HH -> HH-MM (ej. 00-00)
        hora_formatted = f"{hora}:00"

        registros.append({
          "estacion": estacion_actual,
          "fecha": fecha_formatted,
          "hora": hora_formatted,
          "temperatura": float(temp),
          "viento_dir": int(v_dir),
          "viento_vel": int(v_vel),
          "precipitacion": float(precip)
        })
      else:
        # Ignorar encabezados estáticos del SMN
        if (line_str.startswith("*") or 
            "Producto basado" in line_str or 
            "por lo tanto" in line_str or 
            "FECHA" in line_str or 
            "(DIR" in line_str):
            continue

        # Normalizar el nombre de la estación (ej: PILAR_OBS -> PILAR_OBSERVATORIO)
        nombre_estacion = line_str
        if nombre_estacion.endswith("_OBS") or nombre_estacion.endswith(" OBS"):
          nombre_estacion = nombre_estacion[:-4] + "_OBSERVATORIO"
        
        nombre_estacion = nombre_estacion.replace("_", " ").strip()

        posible_estacion = nombre_estacion

  # Crear el DataFrame
  df = pd.DataFrame(registros)

  if not df.empty:
    # Reordenar explícitamente las columnas solicitadas
    columnas = ["estacion", "fecha", "hora", "temperatura", "viento_dir", "viento_vel", "precipitacion"]
    df = df[columnas]
    
    df['precipitacion'] = df["precipitacion"].apply(
      lambda x: 0.0 if x > 150.0 or x < 0 else x
    )

    # Guardar en CSV
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    df.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"✅ CSV generado con éxito en '{ruta_csv}'.")
  else:
    print("⚠️ No se pudieron extraer registros del archivo.")

  return df


# ================================================================
# Datos Meteorologicos horarios
# ================================================================

# WIP....
# Por ahora no se va a usar