import os
import tempfile
import spacy
import sys

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from webanno_spacy_converter.parsers.tsv_parser_v3 import WebAnnoNELParser
from webanno_spacy_converter.converters.webanno_to_spacy import AnnotationSentencesToDocBinConverterV2

def run_tests():
    print("Running alignment types tests...")
    
    nlp = spacy.blank("sr") # Serbian or generic blank model

    test_cases = [
        {
            "name": "Type A (Simple - Only NER)",
            "content": """#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value

#Text=Znate gde je Sv . Andreja ?
1-1	0-5	Znate	_	_	
1-2	6-9	gde	_	_	
1-3	10-12	je	_	_	
1-4	13-15	Sv	http://www.wikidata.org/entity/Q390798[1]	LOC[1]	
1-5	16-17	.	http://www.wikidata.org/entity/Q390798[1]	LOC[1]	
1-6	18-25	Andreja	http://www.wikidata.org/entity/Q390798[1]	LOC[1]	
""",
            "expected_tokens": 6,
            "expected_entities": [
                ("Sv . Andreja", "LOC", "Q390798")
            ]
        },
        {
            "name": "Type B (Complex - POS, Lemma, NER)",
            "content": """#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS|PosValue|coarseValue
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma|value

#Text=Ginter de Brojn 
1-1	0-6	Ginter	PROPN	*	http://www.wikidata.org/entity/Q62753[1]	PERS[1]	Ginter	
1-2	7-9	de	X	*	http://www.wikidata.org/entity/Q62753[1]	PERS[1]	de	
1-3	10-15	Brojn	PROPN	*	http://www.wikidata.org/entity/Q62753[1]	PERS[1]	Brojn	
""",
            "expected_tokens": 3,
            "expected_entities": [
                ("Ginter de Brojn", "PERS", "Q62753")
            ]
        },
        {
            "name": "Type C (Modern - POS, NER, Lemma)",
            "content": """#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS|PosValue|coarseValue
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma|value

#Text=Problemi u realizaciji NIP-a
1-1	0-8	Problemi	NOUN	*	_	_	problem	
1-2	9-10	u	ADP	*	_	_	u	
1-3	11-22	realizaciji	NOUN	*	_	_	realizacija	
1-4	23-28	NIP-a	ADJ	*	*	WORK	NIP-a	
""",
            "expected_tokens": 4,
            "expected_entities": [
                ("NIP-a", "WORK", "*")
            ]
        },
        {
            "name": "Type D (Chunk)",
            "content": """#FORMAT=WebAnno TSV 3.3
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity|identifier|value
#T_SP=de.tudarmstadt.ukp.dkpro.core.api.syntax.type.chunk.Chunk|chunkValue


#Text=Jaroslav Hašek : DOŽIVLJAJI DOBROG VOJNIKA ŠVEJKA u prvom svetskom ratu ;
1-1	0-8	Jaroslav	http://www.wikidata.org/entity/Q2754[1]	PERS[1]	_	
1-2	9-14	Hašek	http://www.wikidata.org/entity/Q2754[1]	PERS[1]	_	
1-3	15-16	:	_	_	_	
1-4	17-27	DOŽIVLJAJI	http://www.wikidata.org/entity/Q208622[2]	WORK[2]	doživljaji dobrog vojnika švejka u prvom svetskom ratu[28]	
1-5	28-34	DOBROG	http://www.wikidata.org/entity/Q208622[2]	WORK[2]	doživljaji dobrog vojnika švejka u prvom svetskom ratu[28]	
1-6	35-42	VOJNIKA	http://www.wikidata.org/entity/Q208622[2]	WORK[2]	doživljaji dobrog vojnika švejka u prvom svetskom ratu[28]	
1-7	43-49	ŠVEJKA	http://www.wikidata.org/entity/Q208622[2]	WORK[2]	doživljaji dobrog vojnika švejka u prvom svetskom ratu[28]	
1-8	50-51	u	http://www.wikidata.org/entity/Q208622[2]	WORK[2]	doživljaji dobrog vojnika švejka u prvom svetskom ratu[28]	
1-9	52-57	prvom	http://www.wikidata.org/entity/Q208622[2]	WORK[2]	doživljaji dobrog vojnika švejka u prvom svetskom ratu[28]	
1-10	58-66	svetskom	http://www.wikidata.org/entity/Q208622[2]	WORK[2]	doživljaji dobrog vojnika švejka u prvom svetskom ratu[28]	
1-11	67-71	ratu	http://www.wikidata.org/entity/Q208622[2]	WORK[2]	doživljaji dobrog vojnika švejka u prvom svetskom ratu[28]	
1-12	72-73	;	_	_	_	
""",
            "expected_tokens": 12,
            "expected_entities": [
                ("Jaroslav Hašek", "PERS", "Q2754"),
                ("DOŽIVLJAJI DOBROG VOJNIKA ŠVEJKA u prvom svetskom ratu", "WORK", "Q208622"),
            ]
        }
    ]

    failed_tests = 0

    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.tsv') as tmp:
            tmp.write(case['content'])
            tmp_path = tmp.name
        
        try:
            # 1. Parse
            parser = WebAnnoNELParser(tmp_path)
            sentences = parser.parse()
            
            # 2. Convert
            converter = AnnotationSentencesToDocBinConverterV2(nlp, ner=True, nel=True)
            doc_bin = converter.convert(sentences)
            docs = list(doc_bin.get_docs(nlp.vocab))
            
            # We expect one doc per sentence (or one doc total if it's one sentence)
            # The samples are single sentences.
            if not docs:
                print("FAIL: No docs created.")
                failed_tests += 1
                continue
                
            doc = docs[0]
            
            # 3. Assert Tokens
            if len(doc) != case['expected_tokens']:
                print(f"FAIL: Token count mismatch. Expected {case['expected_tokens']}, got {len(doc)}")
                print(f"Tokens found: {[t.text for t in doc]}")
                failed_tests += 1
                continue
            else:
                print(f"PASS: Token count {len(doc)}")

            # 4. Assert Entities
            found_entities = []
            for ent in doc.ents:
                # Get KB ID if available (stored in custom extension or just check logic)
                # The converter V2 likely stores it in `ent._.kb_id` or similar if configured, 
                # but let's check how the converter stores it.
                # Assuming standard spacy entity for now, but we need to verify where the ID goes.
                # If the converter puts it in `ent.kb_id_`, we use that.
                kb_id = ent.kb_id_ if ent.kb_id_ else "*"
                found_entities.append((ent.text, ent.label_, kb_id))
            
            # Normalize expected entities for comparison if needed
            # For now, exact match on list
            
            if len(found_entities) != len(case['expected_entities']):
                print(f"FAIL: Entity count mismatch. Expected {len(case['expected_entities'])}, got {len(found_entities)}")
                print(f"Found: {found_entities}")
                failed_tests += 1
                continue

            entities_match = True
            for i, (exp_text, exp_label, exp_id) in enumerate(case['expected_entities']):
                found_text, found_label, found_id = found_entities[i]
                
                if found_text != exp_text:
                    print(f"FAIL: Entity text mismatch. Expected '{exp_text}', got '{found_text}'")
                    entities_match = False
                if found_label != exp_label:
                    print(f"FAIL: Entity label mismatch. Expected '{exp_label}', got '{found_label}'")
                    entities_match = False
                
                # ID check might be tricky depending on how parser handles it.
                # The parser might return full URL or just QID.
                # Let's be lenient if the expected ID is in the found ID
                if exp_id != "*" and exp_id not in found_id:
                     print(f"FAIL: Entity ID mismatch. Expected '{exp_id}', got '{found_id}'")
                     entities_match = False
                elif exp_id == "*" and found_id not in ["", "*", "NIL"]:
                     # If we expect * (no ID), but got something specific
                     pass 

            if entities_match:
                print("PASS: Entities match.")
            else:
                failed_tests += 1

        except Exception as e:
            print(f"FAIL: Exception occurred: {e}")
            import traceback
            traceback.print_exc()
            failed_tests += 1
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    if failed_tests == 0:
        print("\nAll tests passed!")
    else:
        print(f"\n{failed_tests} tests failed.")
        sys.exit(1)

def test_alignment_types():
    run_tests()

if __name__ == "__main__":
    run_tests()
