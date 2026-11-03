# Third-party notices

The root MIT license covers original project contributions. It does not
relicense upstream code or data. The following sources are used or adapted:

| Source | Use here | Upstream terms / attribution |
| --- | --- | --- |
| [AdvBench / llm-attacks](https://github.com/llm-attacks/llm-attacks) | Base harmful requests and derived decompositions | MIT; license retained in `third_party/licenses/advbench.txt` |
| [BABILong](https://github.com/booydar/babilong) | Adapted `babi_utils.py` files and long-context construction approach | Apache-2.0; license retained in `third_party/licenses/babilong.txt`. Utilities were modified to load decomposed AdvBench requests and support this project's context experiments. |
| [PG19](https://github.com/google-deepmind/pg19) | Background text embedded in task archives | Upstream repository Apache-2.0 license retained in `third_party/licenses/pg19.txt`; the corpus contains Project Gutenberg books. Refer to the upstream source for corpus provenance and source-text terms. |
| [HarmBench](https://github.com/centerforaisafety/HarmBench) | HarmBench source behaviors, derived decompositions, and classifier references | MIT; license retained in `third_party/licenses/harmbench.txt` |
| [XSTest](https://github.com/paul-rottger/xstest) | Safe/unsafe prompts and derived over-refusal controls | CC-BY-4.0; license retained in `third_party/licenses/xstest.txt`. Attribution: Paul Röttger and coauthors, *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models*, NAACL 2024. Data was converted to JSON and some prompts were decomposed. |
| [X-Teaming](https://x-teaming.github.io/) | The 1–5 safety-evaluation rubric and judge prompt used by the paper | Attribution: Salman Rahman and coauthors, *X-Teaming: Multi-Turn Jailbreaks and Defenses with Adaptive Multi-Agents*, 2025. No repository-level LICENSE was found in the [official repository](https://github.com/salman-lui/x-teaming) when preparing these notices. The root MIT license does not grant rights to this upstream material; confirm its redistribution terms before public release. |

Copies of retrieved upstream licenses and their URLs are recorded in
`third_party/licenses.json`. This project does not bundle third-party model
weights. Model checkpoints, API services, tokenizers, and dependencies remain
subject to their respective upstream terms.
