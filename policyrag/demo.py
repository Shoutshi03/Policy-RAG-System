from __future__ import annotations

import hashlib
import uuid

from .models import Chunk, Document


def demo_documents() -> list[Document]:
    """Fictional training corpus. It is not an official UNDP publication."""
    records = [
        {
            "name": "DEMO · Résilience climatique (rapport fictif)",
            "language": "fr",
            "pages": 12,
            "passages": [
                (2, "Résumé exécutif", "DOCUMENT FICTIF DE DÉMONSTRATION — Ce scénario pédagogique propose de réduire de 30 % d’ici 2030 les interruptions de services essentiels dans les districts exposés aux inondations et aux sécheresses. La cible est illustrative et ne constitue pas un engagement officiel."),
                (5, "Recommandations · Alerte précoce", "La priorité recommandée est de relier les données météorologiques aux alertes communautaires. Les autorités locales devraient tester des messages d’alerte multilingues, définir des seuils d’évacuation publics et organiser deux exercices par an. Les comités de quartier, les services météorologiques et la protection civile sont les parties prenantes principales."),
                (7, "Financement et mise en œuvre", "Le rapport fictif suggère de réserver une enveloppe de planification équivalente à 2 % des investissements annuels d’infrastructure à l’analyse de risques, à la maintenance préventive et à l’accessibilité des équipements. Cette valeur sert uniquement d’exemple. Les allocations réelles doivent être décidées après une étude budgétaire nationale."),
                (9, "Indicateurs et risques", "Indicateurs proposés : part des ménages couverts par une alerte vérifiée, délai médian entre alerte et diffusion, proportion des centres de santé disposant d’un plan de continuité et nombre de communes ayant cartographié les zones à risque. Risques : couverture réseau inégale, alertes non accessibles aux personnes handicapées et sous-financement de l’entretien."),
                (11, "Inclusion", "Les plans doivent être co-conçus avec les associations de femmes, de jeunes et de personnes handicapées. Les données devraient être ventilées par sexe, âge, handicap et localisation lorsque cela est sûr et pertinent. Les consultations doivent prévoir une compensation des coûts de transport afin d’éviter une participation uniquement symbolique."),
            ],
        },
        {
            "name": "DEMO · Emploi des jeunes (rapport fictif)",
            "language": "en",
            "pages": 10,
            "passages": [
                (1, "Executive summary", "SYNTHETIC DEMONSTRATION DOCUMENT — This fictional policy brief proposes a two-year youth employment pilot combining employer-led apprenticeships, career guidance and small grants for local training providers. Any targets in this sample are illustrative and are not official UNDP findings."),
                (4, "Recommendations", "Recommendation 1: co-design paid apprenticeships with small and medium-sized enterprises. Recommendation 2: publish transparent selection criteria and reserve outreach capacity for young women, rural youth and persons with disabilities. Recommendation 3: connect training completion to recognized credentials and follow-up employment services."),
                (6, "Risks and safeguards", "Key risks include unpaid placements replacing regular jobs, low-quality training, digital exclusion and selection bias. Safeguards should include written learning agreements, grievance channels, periodic employer checks and disaggregated monitoring. The ministry responsible for labour should coordinate with municipalities, training institutions, employer associations and youth organizations."),
                (8, "Monitoring indicators", "Suggested indicators are the share of participants completing training, the share moving into paid work within six months, retention at twelve months, median earnings change and the share of placements meeting decent-work safeguards. Report results by sex, disability, location and income group where consent and data protection allow."),
                (9, "العربية · الشمول", "نموذج توضيحي غير رسمي: ينبغي أن تكون فرص التدريب والعمل متاحة للشابات والشباب في المناطق الريفية وللأشخاص ذوي الإعاقة. يجب نشر معايير اختيار واضحة، وتوفير قناة للشكاوى، وقياس الانتقال إلى العمل اللائق بعد انتهاء التدريب. هذه المعلومات افتراضية لأغراض العرض فقط."),
            ],
        },
        {
            "name": "DEMO · Protection sociale inclusive (rapport fictif)",
            "language": "fr",
            "pages": 14,
            "passages": [
                (3, "Objectifs de politique publique", "DOCUMENT FICTIF DE DÉMONSTRATION — L’objectif du scénario est d’améliorer l’accès aux prestations sociales durant les chocs économiques, sans supposer qu’un registre numérique soit accessible à toutes les personnes. Le dispositif proposé combine guichets locaux, procédures hors ligne et canaux numériques facultatifs."),
                (6, "Options de mise en œuvre", "Option A : élargir temporairement les critères d’éligibilité aux programmes existants. Option B : utiliser des transferts d’urgence avec des règles de sortie publiées. Option C : renforcer les services d’orientation au niveau municipal. Le choix dépend des capacités administratives, du budget et des garanties de recours; les options ne sont pas interchangeables."),
                (8, "Recommandations et garanties", "Recommandations : publier les critères d’admissibilité en langage clair, fournir un mécanisme de plainte accessible, prévoir un réexamen humain des refus automatisés et limiter les données collectées au strict nécessaire. La protection des données et le consentement doivent être documentés avant tout partage de fichiers entre administrations."),
                (10, "Comparaison des approches", "Les transferts d’urgence peuvent être déployés rapidement si les listes sont fiables, mais risquent d’exclure les personnes absentes des registres. Les guichets de proximité peuvent corriger les erreurs mais demandent plus de personnel. Les canaux numériques réduisent certains délais mais doivent toujours être complétés par une solution hors ligne."),
                (12, "Résultats et suivi", "Indicateurs suggérés : délai de versement, part des demandes ayant reçu une décision motivée, taux de recours résolus dans le délai annoncé, taux d’erreur d’exclusion estimé et accessibilité par canal. Un tableau de bord doit distinguer les résultats observés des estimations et publier les limites méthodologiques."),
            ],
        },
    ]

    documents: list[Document] = []
    for record in records:
        digest = hashlib.sha256((record["name"] + "\n" + "\n".join(p[2] for p in record["passages"])).encode()).hexdigest()
        document_id = f"demo-{digest[:16]}"
        chunks = [
            Chunk(
                chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{index}")),
                document_id=document_id,
                file_name=record["name"],
                page=page,
                section=section,
                language=record["language"],
                text=text,
                is_demo=True,
            )
            for index, (page, section, text) in enumerate(record["passages"])
        ]
        documents.append(Document(document_id, record["name"], digest, record["language"], record["pages"], chunks, True))
    return documents
