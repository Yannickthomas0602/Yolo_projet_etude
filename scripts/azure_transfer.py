import os
import time
import json
from datetime import datetime
from typing import Union

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.core.exceptions import ResourceExistsError
except Exception:
    raise RuntimeError("Les packages Azure ne sont pas installés. Installez 'azure-storage-blob' dans votre venv.")

# Configuration depuis l'environnement (.env ou variables système)
AZURE_STORAGE_CONN = os.getenv("AZURE_STORAGE_CONN", "")
AZURE_APPAREIL = os.getenv("AZURE_APPAREIL", "Piscine-Rennes-01")
AZURE_PHOTO_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER", "archives-photos").strip() or "archives-photos"
AZURE_JSON_CONTAINER = os.getenv("AZURE_JSON_CONTAINER", "archives-json").strip() or "archives-json"

if not AZURE_STORAGE_CONN:
    raise RuntimeError("AZURE_STORAGE_CONN non défini. Configurez votre variable d'environnement ou .env")

blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONN)


def ensure_container(container_name: str) -> None:
    """
    Crée le container si possible. Ignore l'erreur si le container existe déjà.
    """
    try:
        blob_service_client.create_container(container_name)
    except ResourceExistsError:
        return
    except Exception:
        # Si la création échoue, on laisse l'upload signaler l'erreur.
        return


def _action_from_status(status: str) -> str:
    normalized_status = (status or "").strip().upper()
    if normalized_status == "BDD":
        return "bon effarouchement"
    if normalized_status == "INCERTITUDE":
        return "enffarouchement par defaut"
    return "pas d'effarouchement"


def transferer_fichiers_azure(
    chemin_image_locale: Union[str, os.PathLike],
    espece_oiseau: str,
    score_confiance: float,
    statut: str,
) -> bool:
    """
    Envoie une image et un JSON descriptif sur Azure Blob Storage.

    - Image -> conteneur `archives-photos`
    - JSON  -> conteneur `archives-json`

    Retourne True si l'opération réussit, False sinon.
    """
    try:
        chemin_image_locale = str(chemin_image_locale)
        if not os.path.exists(chemin_image_locale) or not os.path.isfile(chemin_image_locale):
            print(f"[Azure] Fichier image introuvable: {chemin_image_locale}")
            return False

        timestamp_actuel = int(time.time())
        heure_affichage = datetime.now().strftime("%d/%m/%Y %H:%M")
        # Nom de l'image et du json liés par le même timestamp
        nom_photo = f"detection_{timestamp_actuel}.jpg"
        nom_texte = f"donnees_{timestamp_actuel}.json"

        # 1) Envoi de la photo
        print(f"[Azure] Envoi de l'image {nom_photo} vers '{AZURE_PHOTO_CONTAINER}'...")
        ensure_container(AZURE_PHOTO_CONTAINER)
        blob_photo = blob_service_client.get_blob_client(container=AZURE_PHOTO_CONTAINER, blob=nom_photo)
        with open(chemin_image_locale, "rb") as data:
            blob_photo.upload_blob(data, overwrite=True, content_settings=ContentSettings(content_type="image/jpeg"))

        # 2) Prépare le JSON
        payload = {
            "appareil": AZURE_APPAREIL,
            "oiseau": espece_oiseau,
            "confiance": round(float(score_confiance), 2),
            "action": _action_from_status(statut),
            "image_blob": nom_photo,
            "heure": heure_affichage,
            "timestamp": timestamp_actuel,
            "statut": statut,
        }

        texte_json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

        # 3) Envoi du JSON
        print(f"[Azure] Envoi du fichier JSON {nom_texte} vers '{AZURE_JSON_CONTAINER}'...")
        ensure_container(AZURE_JSON_CONTAINER)
        blob_texte = blob_service_client.get_blob_client(container=AZURE_JSON_CONTAINER, blob=nom_texte)
        blob_texte.upload_blob(texte_json_bytes, overwrite=True, content_settings=ContentSettings(content_type="application/json; charset=utf-8"))

        print("[Azure] Synchronisation réussie : Image + JSON sauvegardés !")
        return True

    except Exception as exc:
        print(f"[Azure Error] Échec de l'envoi : {exc}")
        return False


if __name__ == "__main__":
    # Petit test manuel si exécuté directement
    import argparse

    parser = argparse.ArgumentParser(description="Envoie une image et son JSON descriptif sur Azure Blob")
    parser.add_argument("image", help="Chemin vers l'image locale à envoyer")
    parser.add_argument("espece", help="Espèce détectée (texte)")
    parser.add_argument("score", type=float, help="Score de confiance (0-1)")
    parser.add_argument("statut", nargs="?", default="BDD", help="Statut métier (BDD, INCERTITUDE, HORS_BDD/autre)")
    args = parser.parse_args()

    ok = transferer_fichiers_azure(args.image, args.espece, args.score, args.statut)
    if not ok:
        raise SystemExit(1)
