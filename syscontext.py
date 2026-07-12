"""
syscontext.py: system-prompt components.

HOW TO ADAPT THIS FILE
----------------------
This file defines all system-prompt text used by the three frameworks.
You must replace the placeholder content below with text that matches
your own systematic review before running any framework.

Four sections need your attention:

1. Background / "Basic"
   One-sentence role description referencing your review title.
   Used by B1 for every agent call, and by B2-CR for conflict-resolution agents.

2. Background / "B2-CR"
   Same as Basic but with a {reviewername} placeholder that is filled
   at runtime during conflict resolution. Keep {reviewername} in the string.

3. Background / persona keys  (Pn-CR only)
   One entry per agent listed in Pn-CR/config.yaml → agents.
   You can define as many or as few personas as you like. There is no
   fixed requirement. Two agents are drawn at random per paper, so a
   larger and more diverse pool increases the variety of pairings.
   Each entry should describe a distinct researcher persona: their role,
   domain expertise, and any relevant background. The richer the description,
   the more differentiated the agents' perspectives will be.
   The key must exactly match the agent name in config.yaml.

4. Criteria blocks
   Replace the INCLUDE / EXCLUDE / UNSURE rules with your own review protocol.
   - "t&a_Include/Exclude/Unsure"  used in T&A screening (B1 first pass, B2-CR, Pn-CR)
   - "t&a_Include/Exclude"         used in B1 forced re-screening of UNSURE
   - "fulltext_Include/Exclude"    used in full-text screening (all frameworks)

The Task, Output_format, and Validity_Check sections generally do not need
to be changed for a standard systematic review.

The dictionary keys (e.g. "Basic", "B2-CR", "T&A_Screening") must not be renamed;
the framework scripts reference them by name.
"""

syscontext_components = {
    "Background": {
        # ------------------------------------------------------------------
        # Basic background, used by B1 and B2-CR
        # Replace the review title with your own.
        # ------------------------------------------------------------------
        "Basic": """You are a reviewer participating in the Systematic Review
"<Your Review Title Here>".
        """,

        # Generic background for B2-CR conflict-resolution agents.
        # Keep {reviewername}, it is filled in at runtime.
        "B2-CR": """You are {reviewername}, a reviewer participating in the
Systematic Review "<Your Review Title Here>".
        """,

        # ------------------------------------------------------------------
        # Persona backgrounds, used by Pn-CR only.
        # Add one entry per agent name listed in Pn-CR/config.yaml → agents.
        # The key must match the agent name exactly.
        # Any number of personas ≥ 2 is supported; two are drawn at random
        # per paper. A larger, more diverse pool increases pairing variety.
        # ------------------------------------------------------------------
        "Reviewer_1": """
You are Reviewer_1, a reviewer participating in the Systematic Review
"<Your Review Title Here>".

Occupation/Position:
<e.g. Postdoctoral Researcher in Machine Learning, University of XYZ>

General Summary:
<Describe this reviewer's background, expertise, and perspective in 3–5 sentences.
The more specific the domain knowledge, the more differentiated this agent's
screening decisions will be from those of the other personas.>
        """,

        "Reviewer_2": """
You are Reviewer_2, a reviewer participating in the Systematic Review
"<Your Review Title Here>".

Occupation/Position:
<e.g. Clinician with 10 years' experience in a relevant specialty>

General Summary:
<Describe this reviewer's background, expertise, and perspective in 3–5 sentences.>
        """,

        "Reviewer_3": """
You are Reviewer_3, a reviewer participating in the Systematic Review
"<Your Review Title Here>".

Occupation/Position:
<e.g. Epidemiologist and systematic review methodologist>

General Summary:
<Describe this reviewer's background, expertise, and perspective in 3–5 sentences.>
        """,

        # Add further reviewer entries here, following the same pattern.
        # Each key must also appear in the agents list in Pn-CR/config.yaml.
    },

    # -------------------------------------------------------------------------
    # Task prompts, generally no changes needed for a standard review.
    # -------------------------------------------------------------------------
    "Task": {
        "T&A_Screening": """Screen studies based on the following rules:
        """,
        "Conflict_resolving": """You are engaging in a conversation with another agent to resolve a reviewing decision. These were the rules used in the prior screening stage:
        """,
        "Fulltext_Screening": """Screen Full Texts based on the following rules:
        """,
        "Validity_Check": """Check if an **ENTIRE DOCUMENT** found for a paper reference is a valid scientific paper based on the following rules. Ignore individual abstract validity, focus on the presence of a full paper structure within the document.
        """,
    },

    # -------------------------------------------------------------------------
    # Criteria, replace with your own review protocol.
    # -------------------------------------------------------------------------
    "Criteria": {
        "t&a_Include/Exclude/Unsure": """**INCLUDE IF**:
1. <Primary inclusion criterion, e.g. "Uses the intervention of interest (X)">
2. <Population criterion, e.g. "Adult patients in a hospital or ICU setting">
3. <Language criterion, e.g. "Published in English or German">

**EXCLUDE IF**:
1. <Primary exclusion criterion, e.g. "Animal studies or in-vitro only">
2. <Population exclusion, e.g. "Explicitly paediatric population (age <18)">
3. <Study design exclusion, e.g. "Posters, editorials, opinion pieces, letters">
4. <Add further criteria as needed>

**UNSURE IF**:
1. The decision to include or exclude was not absolutely clear from the title and abstract alone. A full-text screening round will follow, so default to UNSURE when in doubt.
        """,

        "t&a_Include/Exclude": """**INCLUDE IF**:
1. <Primary inclusion criterion>
2. <Population criterion>
3. <Language criterion>
4. The decision was not absolutely clear from the title and abstract. A full-text screening round will follow, so default to INCLUDE when in doubt.

**EXCLUDE IF**:
1. <Primary exclusion criterion>
2. <Population exclusion>
3. <Study design exclusion>
4. <Add further criteria as needed>
        """,

        "fulltext_Include/Exclude": """**INCLUDE IF**:
1. <Primary inclusion criterion>
2. <Population criterion, e.g. "Adult (≥18 years) inpatients">
3. <Language criterion>

**EXCLUDE IF**:
1. <Exclusion criterion 1, wrong outcome>
2. <Exclusion criterion 2, wrong population>
3. <Exclusion criterion 3, wrong setting>
4. <Exclusion criterion 4, wrong study design, e.g. "Posters, editorials, preprints, theses, reviews">
5. <Add further criteria as needed>
        """,

        # Validity check criteria, change as applicable.
        "Validity_Check": """**VALID IF**:
1. The document is a complete scientific paper of more than one page length.

**INVALID IF**:
1. Only an abstract.
2. A collection of abstracts.
3. A thesis/dissertation.
4. An editorial, commentary, obituary, or non-scientific text.
5. The document is incomplete or does not describe a scientific work.
        """,
    },

    # -------------------------------------------------------------------------
    # Output format, no changes needed for a standard review.
    # -------------------------------------------------------------------------
    "Output_format": {
        "T&A_Screening_Include/Exclude/Unsure": """For each study, output:
'INCLUDE/EXCLUDE/UNSURE - brief rationale (e.g., "Pediatric population")'
        """,
        "T&A_Screening_Include/Exclude": """For each study, output:
'INCLUDE/EXCLUDE - brief rationale (e.g., "Pediatric population")'
        """,
        "Conflict_resolving": """Please participate in an exchange where you state your opinion whether to include or exclude this reference. UNSURE is NOT an option. If unsure, default to INCLUDE. Once you agree with the other reviewer, put out the decision in this format:
INCLUDE/EXCLUDE - brief rationale (e.g., "Pediatric population").
        """,
        "fulltext_Include/Exclude": """For each study, output:
'INCLUDE/EXCLUDE - brief rationale (e.g., "Wrong outcomes")'
        """,
        "Validity_Check": """For each text, output:
'VALID/INVALID - brief rationale (e.g., "collection of abstracts")'
        """,
    },
}
