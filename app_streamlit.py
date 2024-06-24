import streamlit as st
import pandas as pd

# Títulos de las pestañas
titulos_pestanas = ['Página principal', 'Datos de entrenamiento', 'Datos de Submission','Sobre nosotras']
pestaña1, pestaña2, pestaña3, pestaña4 = st.tabs(titulos_pestanas)

with pestaña1:
    st.title('Detección de retinopatías diabéticas')
    st.write("La retinopatía diabética es una complicación ocular grave asociada con la diabetes.")
    st.write("Se produce cuando la diabetes afecta los vasos sanguíneos de la retina, la capa de tejido sensible a la luz en la parte posterior del ojo.")
    st.write("Algunos puntos clave:")
    st.write("- Causas: La retinopatía diabética se debe al daño que la diabetes provoca en los vasos sanguíneos de la retina.")
    st.write("  Este daño puede llevar a hemorragias retinianas y pérdida de visión. Es crucial mantener un control efectivo de la diabetes para prevenir o retrasar esta complicación.")
    st.write("- Etapas:")
    st.write("  - No proliferativa (NPDR): En esta etapa inicial, los vasos sanguíneos de la retina muestran signos de daño.")
    st.write("    Puede progresar a la etapa más grave.")
    st.write("  - Proliferativa (PDR): Se forman nuevos vasos sanguíneos anormales en la retina, lo que puede llevar a hemorragias y cicatrices.")
    st.write("    El diagnóstico temprano y el tratamiento son cruciales para prevenir la pérdida de visión.")
    st.write("- Síntomas: La retinopatía diabética a menudo comienza sin síntomas notorios, por lo que es importante realizar exámenes oculares regulares.")
    st.write("  A medida que avanza, pueden aparecer síntomas en el ojo diabético que no deben ignorarse.")
    st.write("En resumen, detectar y tratar la retinopatía diabética a tiempo es fundamental para preservar la visión.")
    st.write("Si tienes diabetes, consulta con un especialista en oftalmología para evaluar tu salud ocular.")

with pestaña2:
    st.header('Datos de Entrenamiento')
    train_df = pd.read_csv.write(train_df)
    

    st.bar_chart(train_df)
    if st.checkbox('Mostrar estadísticas descriptivas de entrenamiento'):
        st.write(train_df.describe())
    
    st.header('Filtrar Datos')
    column_train = st.selectbox('Selecciona una columna para filtrar en el dataset de entrenamiento', train_df.columns)
    value_train = st.text_input(f'Introduce un valor para filtrar en la columna {column_train}', key='value_train')
    if value_train:
        filtered_train_df = train_df[train_df[column_train].astype(str).str.contains(value_train, na=False)]
        st.write(filtered_train_df)

with pestaña3:
    st.header('Datos de Submission')
    submission_df = pd.read_csv('/mnt/data/submissionDR (1).csv')
    st.write(submission_df)
    st.bar_chart(submission_df)
    if st.checkbox('Mostrar estadísticas descriptivas de submission'):
        st.write(submission_df.describe())
    
    st.header('Filtrar Datos')
    column_submission = st.selectbox('Selecciona una columna para filtrar en el dataset de submission', submission_df.columns)
    value_submission = st.text_input(f'Introduce un valor para filtrar en la columna {column_submission}', key='value_submission')
    if value_submission:
        filtered_submission_df = submission_df[submission_df[column_submission].astype(str).str.contains(value_submission, na=False)]
        st.write(filtered_submission_df)

with pestaña4:
    st.title("Sobre nosotras")
    st.write("Información sobre nosotras.")
    st.image("/mnt/data/intro.png")  # Cambia esto por una ruta válida a la imagen
    st.write("[Para más información de click aquí](https://www.kaggle.com/competitions/upch-intro-ml)")

# Ejecutar la aplicación en Streamlit
if __name__ == '__main__':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.title('Visualización de la Base de Datos de Retinopatía')
    st.write('Carga tus archivos CSV para visualizarlos.')


