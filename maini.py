import streamlit as st
from analysedebit import analyser_debit, format_timestamp
from analysepauses import analyser_pauses, generer_export_pauses, graph_pauses

st.title("Analyse du débit de parole d'une vidéo YouTube")
st.markdown("""
Cette application télécharge une vidéo via yt-dlp, découpe un sous-clip et le transcrit avec Whisper.
Choisissez la méthode de segmentation :
- **whisper** : segmentation automatique effectué par Whisper.
- **ponctuation** : découpage de la transcription par ponctuation sur la base des timestamps.
L'analyse inclut l'intégralité du segment défini.
""")

with st.form(key='form_analyse'):
    video_url = st.text_input("URL de la vidéo YouTube", "")
    start_time = st.text_input("Temps de début (en secondes)", "0")
    end_time = st.text_input("Temps de fin (en secondes)", "60")
    repertoire = st.text_input("Répertoire de stockage", "test1")
    segmentation_mode = st.selectbox("Mode de segmentation",
                                     options=["whisper", "ponctuation"],
                                     help="Choisissez la méthode de segmentation.")
    submit_button = st.form_submit_button(label="Analyser la vidéo")

if submit_button:
    try:
        st.info("Analyse en cours...")
        mode = segmentation_mode  # "whisper" ou "ponctuation"

        results = analyser_debit(
            video_url, float(start_time), float(end_time), repertoire,
            segmentation_mode=mode
        )
        (df_debit, chart_barres, chart_combine, export_text, export_fichier,
         df_csv, graph_barres_html, graph_barres_png, graph_combine_html, graph_combine_png, dialogue_file,
         transcript_segments) = results

        st.video(video_url)
        st.subheader("Débit de parole par segment")
        st.dataframe(df_debit)
        st.altair_chart(chart_barres, use_container_width=True)
        st.altair_chart(chart_combine, use_container_width=True)
        st.markdown(f"**DataFrame CSV exportée :** {df_csv}")
        st.markdown(f"**Graphique en barres HTML :** {graph_barres_html}")
        st.markdown(f"**Graphique en barres PNG :** {graph_barres_png}")
        st.markdown(f"**Graphique combiné HTML :** {graph_combine_html}")
        st.markdown(f"**Graphique combiné PNG :** {graph_combine_png}")
        if dialogue_file:
            st.markdown(f"**Fichier de dialogue exporté :** {dialogue_file}")

        st.subheader("Rapport détaillé des segments")
        st.text_area("Export Texte", export_text, height=300)
        st.success(f"Le rapport a été exporté dans : {export_fichier}")

        # Utilisation du module d'analyse des pauses
        pauses = analyser_pauses(transcript_segments, seuil=1.0)
        if pauses:
            pause_text = generer_export_pauses(pauses, format_timestamp)
            st.subheader("Analyse des pauses")
            st.text_area("Pause Analysis", pause_text, height=150)
            chart_pauses = graph_pauses(pauses)
            st.altair_chart(chart_pauses, use_container_width=True)
        else:
            st.subheader("Analyse des pauses")
            st.info("Aucune pause supérieure à 1 seconde n'a été détectée.")

        st.markdown("""
        **Remarque sur le découpage :**
        - En mode **whisper**, seule la segmentation interne détectée par Whisper est utilisée.
        - En mode **ponctuation**, la transcription globale est découpée par ponctuation.
        """)
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")


