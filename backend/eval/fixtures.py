"""
Real, labeled evaluation transcripts for iCall's structuring pipeline.

Pulled verbatim from real completed calls in ithink_dev.db (call_utterances
table) -- not synthetic examples. Each fixture is a full call, turn by turn,
with a human-labeled expected judgment for the discrete signals that matter
most (conflict, corrects_fact, missing_info dedup, is_wrapping_up, and the
not-yet-built root_cause_question/resolution_declared fields) plus a loose
"key terms" check for fact extraction, since exact string match against an
LLM's own phrasing is not a meaningful bar.

This exists so accuracy claims about the structuring prompt (before OR
after any change -- the two-call split, new trigger fields, prompt
tightening) are measured against real, previously-observed speech,
including the exact STT garbling and disfluency these calls actually had --
not against clean, hand-written example sentences that don't reflect what
real audio produces.

Each turn's `expect` dict is what a careful human reader considers correct
given everything said so far in the call, not what any past model run
happened to output. `key_terms_in_facts` is a soft check: at least one new
fact this turn should contain each listed substring (case-insensitive).
"""

FIXTURES = [
    {
        "call_id": 21,
        "channel": "incident-21",
        "description": "Genuine correction: AC region claim retracted, confirmed US East -- real corrects_fact case.",
        "turns": [
            {"speaker": "A", "text": "Yeah. Hi. So we have got an API outage.",
             "expect": {"key_terms_in_facts": ["outage"]}},
            {"speaker": "A", "text": "So the old API service is throwing five times errors across all ports in US East.",
             "expect": {"key_terms_in_facts": ["us east"]}},
            {"speaker": "B", "text": "Yeah. Can you check the payment service logs for the last",
             "expect": {}},  # cut off mid-sentence, nothing clean to extract
            {"speaker": "A", "text": "I think the issue started after yesterday's deploy.",
             "expect": {"hypothesis_expected": True}},
            {"speaker": "A", "text": "Actually confirmed, it's in the AC region",
             "expect": {"key_terms_in_facts": ["ac region"]}},
            {"speaker": "B", "text": "Are you sure? I thought it was US east.",
             "expect": {"conflict_expected": True}},
            {"speaker": "A", "text": "You are right. My mistake confirmed it's US East North AC region.",
             "expect": {"corrects_fact_expected": True, "corrects_fact_should_reference": "AC region"}},
            {"speaker": "B", "text": "Let's roll back yesterday's deploy. That's the safest move.",
             "expect": {"decision_expected": True}},
            {"speaker": "A", "text": "Does anyone know which specific port is affected?",
             "expect": {"missing_info_expected": True}},
            {"speaker": "B", "text": "Okay. Ted covers it. Let's reconvene in thirty minutes.",
             "expect": {"action_item_owner_expected": "Ted"}},
        ],
    },
    {
        "call_id": 32,
        "channel": "incident-39",
        "description": "Root-cause hypothesis raised then explicitly ruled out (resolution declared), a real fact correction (US East -> EU Central), repeated missing-info (cooldown test), direct address, wrap-up.",
        "turns": [
            {"speaker": "A", "text": "Q update.", "expect": {}},
            {"speaker": "A", "text": "There is an outage affecting the payment gateway in the US. East region",
             "expect": {"key_terms_in_facts": ["us east"]}},
            {"speaker": "B", "text": "Think", "expect": {}},
            {"speaker": "B", "text": "it might be a database connection issue on our end.",
             "expect": {"hypothesis_expected": True, "root_cause_question_expected": False}},
            {"speaker": "A", "text": "No. The database team already confirmed everything is working fine on their side.",
             "expect": {"conflict_expected": True, "key_terms_in_facts": ["database"]}},
            {"speaker": "A", "text": "For now, consider it to be ruled out because, there was a confirmation from the team.",
             "expect": {"resolution_declared_expected": True}},
            {"speaker": "B", "text": "Bob", "expect": {}},
            {"speaker": "B", "text": "can you take the roll back and confirm once it's done?",
             "expect": {"action_item_owner_expected": "Bob"}},
            {"speaker": "B", "text": "Actually, it's confirmed the outage is in EU Central. Not US East.",
             "expect": {"corrects_fact_expected": True, "corrects_fact_should_reference": "US East",
                        "key_terms_in_facts": ["eu central"]}},
            {"speaker": "A", "text": "We still don't know is on the call for the database team.",
             "expect": {"missing_info_expected": True}},
            {"speaker": "B", "text": "Yeah.", "expect": {}},
            {"speaker": "B", "text": "Still unclear. Who is on call?",
             "expect": {"missing_info_should_be_deduped": True}},
            {"speaker": "A", "text": "Really? Nobody has confirmed who is on call yet?",
             "expect": {"missing_info_should_be_deduped": True}},
            {"speaker": "A", "text": "Hey.", "expect": {}},
            {"speaker": "A", "text": "Watcher. Can you tell me the status?",
             "expect": {"direct_address_expected": True}},
            {"speaker": "A", "text": "That covers everything. Thanks.",
             "expect": {"is_wrapping_up_expected": True}},
        ],
    },
    {
        "call_id": 19,
        "channel": "incident-26",
        "description": "The exact Africa/AC-region pattern that caused the original false-correction bug: a completion (also in Africa) followed by a genuine correction (not AC region, only Africa) followed by a pure restatement that must NOT re-fire corrects_fact.",
        "turns": [
            {"speaker": "B", "text": "Hi.", "expect": {}},
            {"speaker": "B", "text": "So, basically, there was an outage in the payment", "expect": {}},
            {"speaker": "B", "text": "the AC region.",
             "expect": {"key_terms_in_facts": ["ac region"]}},
            {"speaker": "A", "text": "Yes. Involved late that. Raul heart of the motive.",
             "expect": {}},  # garbled STT -- correct behavior is to extract nothing rather than hallucinate structure
            {"speaker": "A", "text": "Oh, we need to ask it.", "expect": {}},
            {"speaker": "B", "text": "Yeah. So, basically, need to see the things are basically resolved or not. Yeah.",
             "expect": {"resolution_declared_expected": False}},  # a question about resolution, not a declaration of one
            {"speaker": "A", "text": "Yeah. Yeah. So, I request that Rahul joins this meet, and we can have a proper discussion.",
             "expect": {}},  # too vague/social to force a clean action item
            {"speaker": "B", "text": "So, like, I want to state one more", "expect": {}},
            {"speaker": "B", "text": "So I want to say that the outage is also in the Africa region, not just in",
             "expect": {"corrects_fact_expected": False, "key_terms_in_facts": ["africa"]}},  # completion, not contradiction
            {"speaker": "B", "text": "Also, I just got notification from the tech lead CEO that, like, outage is not in the AC region. It's just in the like, Africa region.",
             "expect": {"corrects_fact_expected": True, "corrects_fact_should_reference": "AC region"}},  # genuine reversal
            {"speaker": "B", "text": "Yeah. Yeah.",
             "expect": {"corrects_fact_expected": False}},  # pure restatement -- must not re-fire
            {"speaker": "B", "text": "So", "expect": {}},
            {"speaker": "B", "text": "so, like,", "expect": {}},
        ],
    },
    {
        "call_id": 46,
        "channel": "incident-55",
        "description": "Real correction/conflict collision (2026-09-10): a genuine correction (EU Central -> APAC) got spoken correctly, but a later STT-fragmented turn re-raised the already-resolved region as a new conflict nine seconds later -- confusing, since the room had just heard it settled. Guards the prompt fix telling the model not to reopen something the given facts list already reflects.",
        "turns": [
            {"speaker": "B", "text": "What region is this even in?", "expect": {"missing_info_expected": True}},
            {"speaker": "A", "text": "Hold on. It's the EU Central region.",
             "expect": {"key_terms_in_facts": ["eu central"]}},
            {"speaker": "A", "text": "Hey, actually, I'm saying it's APAC. EU Central was wrong.",
             "expect": {"corrects_fact_expected": True, "corrects_fact_should_reference": "EU Central"}},
            {"speaker": "A", "text": "Yeah, EU Central was wrong too, like I said.",
             "expect": {"conflict_expected": False}},  # facts list already says APAC -- must not reopen as a new conflict
        ],
    },
]
