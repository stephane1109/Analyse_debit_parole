import os
import re
import unicodedata
import subprocess
import pandas as pd
import altair as alt
from yt_dlp import YoutubeDL
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


# =============================================================================
# Fonction utilitaire : formatage d'un timestamp en HH:MM:SS
# =============================================================================
def format_timestamp(seconds: float) -> str:
    """Convertit un temps en secondes en format HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# =============================================================================
# Fonction de nettoyage du nom de fichier
# =============================================================================
def sanitize_filename(filename: str) -> str:
    """
    Nettoie le nom de fichier en retirant accents, espaces et caractères non autorisés.
    """
    nfkd_form = unicodedata.normalize('NFKD', filename)
    ascii_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    ascii_text = ascii_text.replace(" ", "_")
    ascii_text = re.sub(r'[^\w\-_.]', '', ascii_text)
    return ascii_text


# =============================================================================
# Extraction d'un sous-clip vidéo avec ré-encodage (vidéo et audio)
# =============================================================================
def extract_subclip_custom(input_path: str, t1: float, t2: float, output_path: str):
    """
    Extrait un sous-clip de la vidéo entre t1 et t2 (en secondes) en ré-encodant la vidéo et l'audio.
    Cela garantit la présence d'une piste audio.
    """
    commande = [
        "ffmpeg", "-y", "-i", input_path,
        "-ss", str(t1), "-to", str(t2),
        "-c:v", "libx264", "-c:a", "aac", output_path
    ]
    result = subprocess.run(commande, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_message = result.stderr.decode('utf-8')
        raise Exception(f"Erreur lors de l'extraction du sous-clip: {error_message}")


# =============================================================================
# Téléchargement de la vidéo YouTube et renommage
# =============================================================================
def telecharger_video(video_url: str, repertoire: str) -> str:
    """
    Télécharge la vidéo YouTube via yt-dlp et la sauvegarde dans le répertoire spécifié
    avec un nom basé sur le titre (après nettoyage).
    """
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
    options = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(repertoire, '%(title)s.%(ext)s'),
        'quiet': True,
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_url, download=True)
        original_title = info.get('title', 'video')
        extension = info.get('ext', 'mp4')
        original_filename = os.path.join(repertoire, f"{original_title}.{extension}")
        sanitized_title = sanitize_filename(original_title)
        sanitized_filename = os.path.join(repertoire, f"{sanitized_title}.{extension}")
        print("Fichier original :", original_filename)
        print("Fichier sanitizé :", sanitized_filename)
        if original_filename != sanitized_filename and os.path.exists(original_filename):
            os.rename(original_filename, sanitized_filename)
    return sanitized_filename


# =============================================================================
# Transcription par Whisper (mode "whisper")
# =============================================================================
def obtenir_transcription_whisper(video_path: str, start_time: float, end_time: float, repertoire: str):
    """
    Découpe le clip vidéo entre start_time et end_time via extract_subclip_custom et le transcrit avec Whisper.

    Si Whisper retourne des segments internes (résultat['segments'] non vide), on retourne ces segments.
    Sinon, on renvoie un segment global couvrant l'intervalle entier.
    """
    base, ext = os.path.splitext(video_path)
    clip_path = f"{base}_{int(start_time)}_{int(end_time)}{ext}"
    extract_subclip_custom(video_path, start_time, end_time, clip_path)
    try:
        import whisper
    except ImportError:
        raise ImportError("Installez 'openai-whisper' avec pip install -U openai-whisper")
    model = whisper.load_model("small")
    result = model.transcribe(clip_path, language='fr')
    segments = result.get('segments', [])
    if segments and len(segments) > 0:
        return segments
    else:
        full_text = result.get('text', '').strip()
        return [{'start': start_time, 'end': end_time, 'text': full_text}]


# =============================================================================
# Transcription par segmentation ponctuation (mode "ponctuation")
# =============================================================================
def obtenir_transcription_ponctuation(video_path: str, start_time: float, end_time: float, repertoire: str):
    """
    Réalise la transcription globale du clip vidéo entre start_time et end_time avec Whisper,
    découpe le texte par ponctuation en exigeant que la ponctuation soit suivie d'un espace et d'une majuscule,
    répartit uniformément les timestamps sur l'intervalle et filtre les doublons consécutifs.
    Renvoie une liste de segments (dictionnaires) avec 'start', 'end' et 'text'.
    (Le segment global n'est pas ajouté pour éviter la répétition.)
    """
    base, ext = os.path.splitext(video_path)
    clip_path = f"{base}_{int(start_time)}_{int(end_time)}{ext}"
    extract_subclip_custom(video_path, start_time, end_time, clip_path)
    try:
        import whisper
    except ImportError:
        raise ImportError("Installez 'openai-whisper' avec pip install -U openai-whisper")
    model = whisper.load_model("small")
    result = model.transcribe(clip_path, language='fr')
    full_text = result.get('text', '').strip()
    # Découpage en phrases, exigeant que la ponctuation soit suivie d'un espace et d'une majuscule
    phrases = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý])', full_text)
    filtered_phrases = []
    for phrase in phrases:
        phrase = phrase.strip()
        if phrase and (not filtered_phrases or phrase != filtered_phrases[-1]):
            filtered_phrases.append(phrase)
    segments = []
    n = len(filtered_phrases)
    if n == 0:
        segments = [{'start': start_time, 'end': end_time, 'text': full_text}]
    else:
        total_duration = end_time - start_time
        interval = total_duration / n
        for i, phrase in enumerate(filtered_phrases):
            seg_start = start_time + i * interval
            seg_end = seg_start + interval
            segments.append({'start': seg_start, 'end': seg_end, 'text': phrase})
    return segments


# =============================================================================
# Calcul du débit de parole par segment
# =============================================================================
def calculer_debit(transcript_segments):
    """
    Calcule le débit (mots par minute) pour chaque segment.
    """
    debits = []
    for segment in transcript_segments:
        debut = float(segment['start'])
        fin = float(segment.get('end', debut))
        texte = segment['text']
        nb_mots = len(texte.split())
        duree = fin - debut if fin - debut > 0 else 1
        debit = (nb_mots / duree) * 60
        debits.append({
            'start': debut,
            'end': fin,
            'nb_mots': nb_mots,
            'duree': round(duree, 2),
            'debit_mpm': round(debit, 2)
        })
    return debits


# =============================================================================
# Calcul du débit global moyen
# =============================================================================
def calculer_debit_global(transcript_segments):
    """
    Calcule le débit global moyen (mots par minute) sur tous les segments.
    """
    total_mots = sum(len(seg['text'].split()) for seg in transcript_segments)
    total_duree = sum(float(seg.get('end', seg['start'])) - float(seg['start']) for seg in transcript_segments)
    debit_global = (total_mots / total_duree * 60) if total_duree > 0 else 0
    return debit_global


# =============================================================================
# Export de la DataFrame en CSV
# =============================================================================
def exporter_dataframe(df: pd.DataFrame, repertoire: str, nom_fichier="segments.csv"):
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
    chemin = os.path.join(repertoire, nom_fichier)
    df.to_csv(chemin, index=False)
    return chemin


# =============================================================================
# Export d'un graphique Altair en HTML
# =============================================================================
def exporter_graphique_html(chart: alt.Chart, repertoire: str, nom_fichier="graphique.html"):
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
    chemin = os.path.join(repertoire, nom_fichier)
    chart.save(chemin)
    return chemin


# =============================================================================
# Export d'un graphique Altair en PNG
# =============================================================================
def exporter_graphique_png(chart: alt.Chart, repertoire: str, nom_fichier="graphique.png"):
    """
    Exporte un graphique Altair en PNG.
    Nécessite 'vl-convert-python' (pip install vl-convert-python).
    """
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
    chemin = os.path.join(repertoire, nom_fichier)
    chart.save(chemin, scale_factor=2.0)
    return chemin


# =============================================================================
# Génération du rapport texte avec timestamps formatés
# =============================================================================
def generer_export_texte(transcript_segments, debit_global):
    """
    Génère un rapport texte détaillé pour chaque segment, avec timestamps en HH:MM:SS.
    """
    lignes = []
    for i, seg in enumerate(transcript_segments, start=1):
        debut = float(seg['start'])
        fin = float(seg.get('end', debut))
        texte = seg['text'].strip()
        nb_mots = len(texte.split())
        duree = fin - debut if fin - debut > 0 else 1
        debit = (nb_mots / duree) * 60
        lignes.append(
            f"Segment {i}: {format_timestamp(debut)} ({debut:.2f}s) à {format_timestamp(fin)} ({fin:.2f}s), {nb_mots} mots, débit = {debit:.2f} mots/min")
        lignes.append("Transcription:")
        phrases = re.split(r'(?<=[.!?])\s+', texte)
        for phrase in phrases:
            if phrase:
                lignes.append(f"  - {phrase}")
        lignes.append("")
    lignes.append(f"Débit global moyen = {debit_global:.2f} mots/min")
    return "\n".join(lignes)


# =============================================================================
# Export du rapport texte dans un fichier
# =============================================================================
def exporter_rapport(export_text, repertoire, nom_fichier="rapport_transcription.txt"):
    if not os.path.exists(repertoire):
        os.makedirs(repertoire)
    chemin = os.path.join(repertoire, nom_fichier)
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(export_text)
    return chemin


# =============================================================================
# Fonction principale d'analyse du débit de parole
# =============================================================================
def analyser_debit(video_url, start_time, end_time, repertoire,
                   segmentation_mode="whisper"):
    """
    Analyse complète du débit de parole pour une vidéo YouTube en utilisant Whisper.

    Modes disponibles :
      - "whisper": segmentation automatique par Whisper (utilise les segments internes détectés).
      - "ponctuation": découpage de la transcription globale par ponctuation avec répartition des timestamps.

    Le segment global (défini par start et end) n'est PAS inséré dans le mode "whisper" ni "ponctuation"
    pour éviter la répétition.

    Renvoie :
      - df_debit : DataFrame des segments.
      - chart_barres : Graphique en barres (avec ligne rouge pour la moyenne).
      - chart_combine : Graphique en ligne combiné (courbe + ligne du débit global).
      - export_text : Rapport texte détaillé.
      - export_fichier : Chemin du rapport texte exporté.
      - df_csv : Chemin du CSV exporté.
      - graph_barres_html : Chemin du graphique en barres HTML.
      - graph_barres_png : Chemin du graphique en barres PNG.
      - graph_combine_html : Chemin du graphique combiné HTML.
      - graph_combine_png : Chemin du graphique combiné PNG.
      - dialogue_file : Chemin du fichier "dialogues.txt" (pour le mode ponctuation, sinon vide).
    """
    video_path = telecharger_video(video_url, repertoire)
    dialogue_file = ""
    if segmentation_mode == "whisper":
        transcript_segments = obtenir_transcription_whisper(video_path, start_time, end_time, repertoire)
        segments_plot = transcript_segments
    elif segmentation_mode == "ponctuation":
        transcript_segments = obtenir_transcription_ponctuation(video_path, start_time, end_time, repertoire)
        segments_plot = transcript_segments
        # Export du dialogue global pour vérification (mode ponctuation)
        global_transcript = obtenir_transcription_whisper(video_path, start_time, end_time, repertoire)[0]['text']
        dialogue_file = exporter_rapport(global_transcript, repertoire, nom_fichier="dialogues.txt")
    else:
        raise ValueError("Mode de segmentation non supporté.")

    debits = calculer_debit(segments_plot)
    df_debit = pd.DataFrame(debits)
    debit_global = calculer_debit_global(transcript_segments)

    # Graphique en barres avec ligne rouge pour la moyenne
    chart_barres = alt.Chart(df_debit).mark_bar().encode(
        x=alt.X('start:Q',
                title='Début du segment (s)',
                scale=alt.Scale(domain=[0, df_debit['start'].max()])),
        y=alt.Y('debit_mpm:Q',
                axis=alt.Axis(format=".2f", title='Débit (mots/min)')),
        tooltip=[
            alt.Tooltip('start:Q', format=".2f", title='Start (s)'),
            alt.Tooltip('end:Q', format=".2f", title='End (s)'),
            alt.Tooltip('nb_mots:Q', format=".2f", title='Nb Mots'),
            alt.Tooltip('debit_mpm:Q', format=".2f", title='Débit')
        ]
    ).properties(
        title="Débit de parole par segment (barres)",
        width=600,
        height=300
    )

    # Ajout d'une règle rouge indiquant la moyenne (débit global)
    rule = alt.Chart(pd.DataFrame({
        'start': [0, df_debit['start'].max()],
        'avg': [debit_global, debit_global]
    })).mark_rule(color='red').encode(
        y=alt.Y('avg:Q', axis=alt.Axis(format=".2f"))
    )
    chart_barres = chart_barres + rule

    # Graphique en ligne avec point et règle rouge pour le débit global
    chart_ligne = alt.Chart(df_debit).mark_line(point=True).encode(
        x=alt.X('start:Q', title='Début du segment (s)'),
        y=alt.Y('debit_mpm:Q',
                axis=alt.Axis(format=".2f", title='Débit (mots/min)')),
        tooltip=[
            alt.Tooltip('start:Q', format=".2f", title='Start (s)'),
            alt.Tooltip('end:Q', format=".2f", title='End (s)'),
            alt.Tooltip('nb_mots:Q', format=".2f", title='Nb Mots'),
            alt.Tooltip('debit_mpm:Q', format=".2f", title='Débit')
        ]
    )
    moyenne_df = pd.DataFrame({
        'start': [0, df_debit['start'].max()],
        'debit_global': [debit_global, debit_global]
    })
    ligne_moyenne = alt.Chart(moyenne_df).mark_rule(color='red').encode(
        y=alt.Y('debit_global:Q', axis=alt.Axis(format=".2f"))
    )
    chart_combine = (chart_ligne + ligne_moyenne).properties(
        title="Évolution du débit avec débit global moyen",
        width=600,
        height=300
    )

    export_text = generer_export_texte(transcript_segments, debit_global)
    export_fichier = exporter_rapport(export_text, repertoire, nom_fichier="rapport_transcription.txt")
    df_csv = exporter_dataframe(df_debit, repertoire, nom_fichier="segments.csv")
    graph_barres_html = exporter_graphique_html(chart_barres, repertoire, nom_fichier="chart_barres.html")
    graph_barres_png = exporter_graphique_png(chart_barres, repertoire, nom_fichier="chart_barres.png")
    graph_combine_html = exporter_graphique_html(chart_combine, repertoire, nom_fichier="chart_combine.html")
    graph_combine_png = exporter_graphique_png(chart_combine, repertoire, nom_fichier="chart_combine.png")

    return (df_debit, chart_barres, chart_combine, export_text, export_fichier,
            df_csv, graph_barres_html, graph_barres_png, graph_combine_html, graph_combine_png, dialogue_file,
            transcript_segments)
