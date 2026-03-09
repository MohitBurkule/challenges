#!/usr/bin/env python3
"""
Keyword-based multilingual intent classification with German/French support.

This approach uses keyword matching to help with cross-lingual transfer
where XLM-RoBERTa fails on German/French text.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("public")
OUTPUT_DIR = Path("outputs")

# Multilingual keyword mappings for common intents
INTENT_KEYWORDS = {
    # Cancel-related
    'cancel_order': {
        'de': ['stornier', 'abbestell', 'aufheb', 'beenden'],
        'fr': ['annuler', 'commande'],
        'en': ['cancel', 'order'],
    },
    'cancel_subscription': {
        'de': ['abonnement', 'künd', 'beenden', 'abbestell'],
        'fr': ['abonnement', 'résili', 'annuler'],
        'en': ['subscription', 'cancel'],
    },
    'cancel_appointment': {
        'de': ['termin', 'absag', 'stornier'],
        'fr': ['rendez-vous', 'annuler', 'rdv'],
        'en': ['appointment', 'cancel'],
    },

    # Account-related
    'create_account': {
        'de': ['konto erstell', 'anmeld', 'registrier', 'konto eröffnen'],
        'fr': ['créer compte', 'inscription', 'nouveau compte'],
        'en': ['create account', 'sign up', 'register'],
    },
    'delete_account': {
        'de': ['konto lösch', 'konto entfernen', 'konto schließen'],
        'fr': ['supprimer compte', 'effacer compte', 'fermer compte'],
        'en': ['delete account', 'remove account', 'close account'],
    },
    'change_password': {
        'de': ['passwort änder', 'passwort wechsel', 'kennwort'],
        'fr': ['changer mot de passe', 'modifier mot de passe'],
        'en': ['change password', 'update password'],
    },
    'reset_password': {
        'de': ['passwort zurücksetz', 'passwort vergess', 'neues passwort'],
        'fr': ['réinitialiser mot de passe', 'oublié mot de passe'],
        'en': ['reset password', 'forgot password'],
    },

    # Order-related
    'place_order': {
        'de': ['bestell', 'kauf', 'ordern', 'order'],
        'fr': ['commander', 'acheter', 'passer commande'],
        'en': ['place order', 'buy', 'order'],
    },
    'track_order': {
        'de': ['verfolg', 'status', 'wo ist', 'lieferung'],
        'fr': ['suivi', 'suivre commande', 'où est'],
        'en': ['track', 'where is', 'delivery status'],
    },
    'check_order_status': {
        'de': ['bestellstatus', 'status der bestellung', 'meine bestellung'],
        'fr': ['statut commande', 'état commande'],
        'en': ['order status', 'check order'],
    },
    'modify_order': {
        'de': ['bestellung änder', 'ändern', 'ändern bestell'],
        'fr': ['modifier commande', 'changer commande'],
        'en': ['modify order', 'change order', 'edit order'],
    },

    # Refund-related
    'request_refund': {
        'de': ['rückerstatt', 'geld zurück', 'erstattung'],
        'fr': ['remboursement', 'rembourser', 'argent'],
        'en': ['refund', 'money back'],
    },

    # Support-related
    'contact_support': {
        'de': ['support', 'hilfe', 'kundendienst', 'kontakt'],
        'fr': ['support', 'aide', 'service client', 'contacter'],
        'en': ['support', 'help', 'contact'],
    },
    'speak_to_human': {
        'de': ['mit mensch', 'sprech', 'agent', 'mitarbeiter'],
        'fr': ['parler humain', 'agent', 'personne'],
        'en': ['speak human', 'agent', 'representative'],
    },

    # Product-related
    'check_availability': {
        'de': ['verfügbar', 'lieferbar', 'vorhanden', 'auf lager'],
        'fr': ['disponible', 'disponibilité', 'en stock'],
        'en': ['available', 'availability', 'in stock'],
    },
    'check_stock': {
        'de': ['lagerbestand', 'vorrat', 'bestand'],
        'fr': ['stock', 'réserve'],
        'en': ['stock', 'inventory'],
    },

    # Time-related
    'ask_business_hours': {
        'de': ['öffnungszeit', 'geöffnet', 'geschäftszeit', 'uhrzeit'],
        'fr': ["heure d'ouverture", 'ouvert', 'horaire'],
        'en': ['business hours', 'open', 'hours'],
    },
    'time_request': {
        'de': ['uhrzeit', 'zeit', 'wann'],
        'fr': ['heure', 'temps', 'quand'],
        'en': ['time', 'when', 'what time'],
    },

    # Returns
    'return_item': {
        'de': ['zurück', 'retoure', 'rücksand', 'widerruf'],
        'fr': ['retour', 'renvoyer', 'article'],
        'en': ['return', 'send back'],
    },
    'exchange_item': {
        'de': ['tauschen', 'umtausch', 'austauschen'],
        'fr': ['échanger', 'échange'],
        'en': ['exchange', 'swap'],
    },

    # Payment
    'report_payment_failure': {
        'de': ['zahlung fehlgeschlagen', 'bezahlung nicht', 'zahlungsfehler'],
        'fr': ['échec paiement', 'paiement refusé', 'problème paiement'],
        'en': ['payment failed', 'payment failure', 'declined'],
    },
    'update_payment_method': {
        'de': ['zahlungsmethode änder', 'zahlungsmittel', 'neue zahlung'],
        'fr': ['moyen paiement', 'modifier paiement'],
        'en': ['payment method', 'update payment', 'change payment'],
    },

    # Subscription
    'upgrade_subscription': {
        'de': ['upgrade', 'verbessern', 'höheres abo'],
        'fr': ['améliorer', 'upgrade', 'supérieur'],
        'en': ['upgrade', 'better plan'],
    },
    'downgrade_subscription': {
        'de': ['downgrade', 'verringern', 'geringer'],
        'fr': ['réduire', 'downgrade'],
        'en': ['downgrade', 'reduce'],
    },
    'reactivate_subscription': {
        'de': ['reaktivieren', 'wieder aktiv', 'erneuern'],
        'fr': ['réactiver', 'renouveler'],
        'en': ['reactivate', 'renew', 'restart'],
    },

    # General help
    'ask_about_pricing': {
        'de': ['preis', 'kosten', 'wieviel kostet', 'tarif'],
        'fr': ['prix', 'coût', 'combien', 'tarif'],
        'en': ['price', 'cost', 'how much', 'pricing'],
    },
    'ask_about_products': {
        'de': ['produkt', 'artikel', 'ware', 'produkte'],
        'fr': ['produit', 'article', 'marchandise'],
        'en': ['product', 'item', 'merchandise'],
    },

    # Technical
    'request_software_update': {
        'de': ['software update', 'aktualisier', 'update'],
        'fr': ['mise à jour', 'actualiser', 'update'],
        'en': ['software update', 'update'],
    },
    'troubleshoot_issue': {
        'de': ['fehler', 'problem', 'funktioniert nicht', 'kaputt'],
        'fr': ['problème', 'bug', 'ne fonctionne pas', 'panne'],
        'en': ['troubleshoot', 'issue', 'problem', 'not working'],
    },
    'report_bug': {
        'de': ['bug melden', 'fehler melden', 'fehlerbericht'],
        'fr': ['signaler bug', 'rapporter erreur'],
        'en': ['report bug', 'bug'],
    },

    # Appointments
    'book_appointment': {
        'de': ['termin buchen', 'vereinbaren', 'termin machen'],
        'fr': ['prendre rendez-vous', 'réserver', 'rdv'],
        'en': ['book appointment', 'schedule', 'appointment'],
    },
    'reschedule_appointment': {
        'de': ['termin verschieben', 'ändern termin', 'neuer termin'],
        'fr': ['reporter rendez-vous', 'changer rdv'],
        'en': ['reschedule', 'change appointment'],
    },

    # Account info
    'check_invoice': {
        'de': ['rechnung', 'quittung', 'beleg'],
        'fr': ['facture', 'reçu'],
        'en': ['invoice', 'receipt', 'bill'],
    },
    'check_balance': {
        'de': ['kontostand', 'guthaben', 'saldo'],
        'fr': ['solde', 'balance'],
        'en': ['balance', 'account balance'],
    },

    # Settings
    'change_language': {
        'de': ['sprache ändern', 'sprache wechseln'],
        'fr': ['changer langue', 'modifier langue'],
        'en': ['change language', 'switch language'],
    },
    'update_contact_info': {
        'de': ['kontaktinformationen', 'adresse ändern', 'telefonnummer'],
        'fr': ['informations contact', 'changer adresse'],
        'en': ['contact info', 'update address', 'phone number'],
    },

    # Delivery
    'change_delivery_address': {
        'de': ['lieferadresse', 'adresse ändern', 'lieferort'],
        'fr': ['adresse livraison', 'changer adresse'],
        'en': ['delivery address', 'shipping address'],
    },

    # General
    'ask_for_help': {
        'de': ['hilfe', 'hilf', 'unterstützung', 'bitte helfen'],
        'fr': ['aide', "j'ai besoin d'aide", 'aider moi'],
        'en': ['help', 'need help', 'assist'],
    },
    'ask_about_policies': {
        'de': ['richtlinie', 'politik', 'regel', 'bedingung'],
        'fr': ['politique', 'règle', 'condition'],
        'en': ['policy', 'policies', 'rules'],
    },

    # Logout
    'logout_request': {
        'de': ['abmelden', 'ausloggen', 'logout'],
        'fr': ['déconnexion', 'se déconnecter'],
        'en': ['logout', 'sign out', 'log out'],
    },

    # Login
    'login_request': {
        'de': ['anmelden', 'einloggen', 'login'],
        'fr': ['connexion', 'se connecter'],
        'en': ['login', 'sign in', 'log in'],
    },

    # Demo
    'request_demo': {
        'de': ['demo', 'vorführung', 'test'],
        'fr': ['démonstration', 'demo', 'essai'],
        'en': ['demo', 'demonstration', 'trial'],
    },

    # Account settings
    'configure_settings': {
        'de': ['einstellungen', 'konfigurieren', 'optionen'],
        'fr': ['paramètres', 'configuration', 'options'],
        'en': ['settings', 'configure', 'options'],
    },

    # Feedback
    'rate_service': {
        'de': ['bewerten', 'feedback', 'rezension'],
        'fr': ['évaluer', 'note', 'avis'],
        'en': ['rate', 'rating', 'feedback'],
    },
    'submit_feedback': {
        'de': ['feedback geben', 'rückmeldung', 'kommentar'],
        'fr': ['donner avis', 'retour', 'commentaire'],
        'en': ['feedback', 'comment', 'review'],
    },

    # Delete data
    'request_data_deletion': {
        'de': ['daten löschen', 'löschen daten', 'entfernen'],
        'fr': ['supprimer données', 'effacer données'],
        'en': ['delete data', 'remove data'],
    },

    # Check warranty
    'check_warranty': {
        'de': ['garantie', 'gewährleistung', 'garantiezeit'],
        'fr': ['garantie', 'garant'],
        'en': ['warranty', 'guarantee'],
    },

    # Request catalog
    'request_catalog': {
        'de': ['katalog', 'broschüre', 'prospekt'],
        'fr': ['catalogue', 'brochure'],
        'en': ['catalog', 'brochure'],
    },

    # Request callback
    'request_callback': {
        'de': ['rückruf', 'anrufen', 'zurückrufen'],
        'fr': ['rappel', 'appeler'],
        'en': ['callback', 'call back', 'phone'],
    },

    # Confirm
    'confirm_appointment': {
        'de': ['bestätigen', 'termin bestätig'],
        'fr': ['confirmer', 'confirmation'],
        'en': ['confirm', 'confirmation'],
    },

    # Request info
    'ask_contact_info': {
        'de': ['kontaktinfo', 'erreichen', 'adresse'],
        'fr': ['contact', 'coordonnées'],
        'en': ['contact info', 'reach'],
    },

    # Set reminder
    'set_reminder': {
        'de': ['erinnerung', 'erinnern', 'alarm'],
        'fr': ['rappel', 'rappeler'],
        'en': ['reminder', 'remind'],
    },

    # Request invoice
    'request_invoice': {
        'de': ['rechnung anforder', 'quittung', 'beleg'],
        'fr': ['demander facture', 'facture'],
        'en': ['request invoice', 'invoice'],
    },

    # Improve performance
    'improve_performance': {
        'de': ['leistung', 'verbessern', 'schneller'],
        'fr': ['performance', 'améliorer', 'accélérer'],
        'en': ['performance', 'improve', 'faster'],
    },

    # Report issue
    'report_issue': {
        'de': ['problem melden', 'fehler', 'beschwerde'],
        'fr': ['signaler problème', 'réclamation'],
        'en': ['report issue', 'complaint'],
    },

    # Check billing cycle
    'check_billing_cycle': {
        'de': ['abrechnungszeitraum', 'rechnungszyklus', 'billing'],
        'fr': ['cycle facturation', 'période facturation'],
        'en': ['billing cycle', 'billing period'],
    },

    # Check usage
    'check_usage': {
        'de': ['nutzung', 'verwendung', 'verbrauch'],
        'fr': ['utilisation', 'usage'],
        'en': ['usage', 'consumption'],
    },

    # Delete account
    'delete_account': {
        'de': ['konto löschen', 'konto entfernen'],
        'fr': ['supprimer compte', 'fermer compte'],
        'en': ['delete account', 'close account'],
    },

    # Data usage
    'ask_about_data_usage': {
        'de': ['datennutzung', 'datenverwendung', 'daten wie'],
        'fr': ['utilisation données', 'données comment'],
        'en': ['data usage', 'how data used'],
    },

    # Reset device
    'reset_device': {
        'de': ['gerät zurücksetzen', 'reset', 'neustart'],
        'fr': ['réinitialiser appareil', 'reset'],
        'en': ['reset device', 'factory reset'],
    },

    # Upgrade account
    'upgrade_account': {
        'de': ['konto upgraden', 'verbessern', 'premium'],
        'fr': ['améliorer compte', 'premium'],
        'en': ['upgrade account', 'premium'],
    },

    # Switch plan
    'switch_plan': {
        'de': ['plan wechseln', 'tarif ändern', 'abo wechsel'],
        'fr': ['changer forfait', 'changer plan'],
        'en': ['switch plan', 'change plan'],
    },

    # Get directions
    'get_directions': {
        'de': ['wegbeschreibung', 'anfahrt', 'richtung'],
        'fr': ['itinéraire', 'direction', 'comment aller'],
        'en': ['directions', 'how to get', 'route'],
    },

    # Track shipment
    'track_shipment': {
        'de': ['sendung verfolgen', 'paket', 'lieferung'],
        'fr': ['suivre colis', 'expédition'],
        'en': ['track shipment', 'package'],
    },

    # Transfer
    'request_transfer': {
        'de': ['transfer', 'überweisen', 'übertrag'],
        'fr': ['transfert', 'virer'],
        'en': ['transfer', 'move'],
    },

    # Check status
    'check_status': {
        'de': ['status', 'zustand', 'stand'],
        'fr': ['statut', 'état'],
        'en': ['status', 'check status'],
    },

    # Unsubscribe
    'unsubscribe': {
        'de': ['abmelden', 'abbestellen', 'newsletter'],
        'fr': ['désabonner', 'désinscription'],
        'en': ['unsubscribe', 'opt out'],
    },
}


def score_intent(text: str, language: str, intent: str) -> float:
    """Score how well text matches an intent using keywords."""
    text_lower = text.lower()
    score = 0.0

    if intent not in INTENT_KEYWORDS:
        return 0.0

    keywords = INTENT_KEYWORDS[intent]

    # Score based on language
    if language in keywords:
        for kw in keywords[language]:
            if kw in text_lower:
                score += 2.0  # High weight for language-specific match

    # Also check English keywords (often work cross-lingually)
    if 'en' in keywords:
        for kw in keywords['en']:
            if kw in text_lower:
                score += 1.0  # Lower weight for English keyword

    return score


def classify_with_keywords(text: str, language: str, valid_intents: list) -> tuple:
    """Classify text using keyword matching."""
    scores = {}

    for intent in valid_intents:
        scores[intent] = score_intent(text, language, intent)

    if max(scores.values()) > 0:
        best_intent = max(scores, key=scores.get)
        return best_intent, scores[best_intent]

    return None, 0.0


def main():
    print("=" * 60)
    print("Keyword-based Multilingual Intent Classification")
    print("=" * 60)

    # Load data
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    valid_intents = sorted(train_df["label"].unique())
    print(f"Valid intents: {len(valid_intents)}")

    # Evaluate on training sample
    print("\nEvaluating on training sample...")
    sample_df = train_df.sample(n=500, random_state=42)

    correct = 0
    total = 0
    for _, row in sample_df.iterrows():
        pred, score = classify_with_keywords(row["text"], row["language"], valid_intents)
        if pred == row["label"]:
            correct += 1
        total += 1

    print(f"Training sample accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # Load XLM-R predictions
    xlmr_df = pd.read_csv(OUTPUT_DIR / "submission_xlmr.csv")

    # Create ensemble: use keywords when confidence is high
    print("\nCreating ensemble predictions...")
    ensemble_preds = []
    keyword_only_preds = []

    for idx, row in test_df.iterrows():
        text = row["text"]
        lang = row["language"]
        xlmr_pred = xlmr_df[xlmr_df["id"] == row["id"]]["label"].values[0]

        kw_pred, kw_score = classify_with_keywords(text, lang, valid_intents)

        keyword_only_preds.append(kw_pred if kw_pred else xlmr_pred)

        # Ensemble logic:
        # - If keyword score is high (>3), prefer keyword
        # - Otherwise use XLM-R
        if kw_score > 3.0:
            ensemble_preds.append(kw_pred)
        else:
            ensemble_preds.append(xlmr_pred)

    # Save submissions
    pd.DataFrame({"id": test_df["id"], "label": keyword_only_preds}).to_csv(
        OUTPUT_DIR / "submission_keywords.csv", index=False
    )
    pd.DataFrame({"id": test_df["id"], "label": ensemble_preds}).to_csv(
        OUTPUT_DIR / "submission_ensemble_keywords.csv", index=False
    )

    # Compare
    agreement = sum(1 for x, e in zip(xlmr_df["label"], ensemble_preds) if x == e)
    print(f"\nXLM-R vs Ensemble agreement: {agreement}/{len(test_df)} ({100*agreement/len(test_df):.1f}%)")

    # Save best submission
    import shutil
    shutil.copy(OUTPUT_DIR / "submission_ensemble_keywords.csv", "submission.csv")
    print(f"\nSaved: submission.csv (ensemble with keywords)")

    print("\n" + "=" * 60)
    print("Done! Submissions:")
    print(f"  - outputs/submission_keywords.csv (keyword only)")
    print(f"  - outputs/submission_ensemble_keywords.csv (XLM-R + keywords)")
    print("=" * 60)


if __name__ == "__main__":
    main()
