export const meta = {
  name: 'chi-review-venue',
  description: 'Venue-appropriate review: Paper A as an NLP/COLM submission, Paper B as a CHI/CSCW submission',
  phases: [{ title: 'ReviewA-NLP' }, { title: 'ReviewB-HCI' }],
}
const PAPERS={A:'/home/carlostoxtli/jupyterlab/judge/reviews/paper_A_for_review.txt',
              B:'/home/carlostoxtli/jupyterlab/judge/reviews/paper_B_for_review.txt'}
const SCHEMA={ type:'object', additionalProperties:false, properties:{
    summary:{type:'string'}, strengths:{type:'array',items:{type:'string'}},
    weaknesses:{type:'array',items:{type:'string'}},
    required_revisions:{type:'array',items:{type:'string'}},
    requested_experiments:{type:'array',items:{type:'string'}},
    overall_rating:{type:'number'}, recommendation:{type:'string',enum:['reject','major revision','minor revision','accept']},
    confidence:{type:'integer'} },
  required:['summary','strengths','weaknesses','required_revisions','requested_experiments','overall_rating','recommendation','confidence'] }
const A_RUBRIC=`You are an independent expert reviewer for COLM / EMNLP (top-tier NLP venues), reviewing this as an
“LLM-as-a-judge / automatic-evaluation” submission. You have ONLY this paper, no other reviews or context. Judge it by
NLP-venue norms: soundness of the meta-evaluation methodology, fair comparison and uncertainty, value to the
automatic-evaluation community, related work in NLP evaluation, reproducibility. NOTE: an open-models-only scope,
classification/NLI tasks, and the absence of an HCI/user-study contribution are NORMAL and appropriate for this venue,
not weaknesses; do not penalize the paper for not being an HCI paper or for not running proprietary models if the
open-model scope is clearly stated. Use the 1.0-5.0 scale (3.0 borderline). In requested_experiments list only
concretely runnable additions. Be rigorous but venue-appropriate.`
const B_RUBRIC=`You are an independent expert reviewer for ACM CHI / CSCW (top-tier HCI venues), reviewing this as a
Human-Centered Computing submission that pairs a systematic review with an empirical study. You have ONLY this paper,
no other reviews or context. Judge by CHI/CSCW norms: contribution to HCC, the review's rigor and the empirical
study's soundness, epistemological framing, ethics, and implications for practice. NOTE: an open-models-only scope is
a reasonable, reproducible choice; the human ground truth is inherited from the datasets' own annotators (no extra
human IRR is required for the empirical claims). Use the 1.0-5.0 scale (3.0 borderline). In requested_experiments
list only concretely runnable additions. Be rigorous but venue-appropriate.`
const A_lenses=[{id:'R1',a:'LLM-as-a-judge meta-evaluation & agreement metrics'},{id:'R2',a:'NLP automatic evaluation & benchmarking'},
                {id:'R3',a:'statistics & experimental design for NLP'},{id:'R4',a:'reproducibility, contamination & generalization in NLP'}]
const B_lenses=[{id:'R1',a:'HCI / qualitative methods & research epistemology'},{id:'R2',a:'CSCW / social computing & crowd annotation'},
                {id:'R3',a:'statistics & measurement for HCI'},{id:'R4',a:'responsible AI & research ethics in HCI'}]
async function panel(key,rubric,lenses,phase){
  return parallel(lenses.map(L=>()=>
    agent(`${rubric}\n\nYour expertise lens: ${L.a}.\n\nRead the full paper here and review it:\n${PAPERS[key]}\n\nReturn your structured review.`,
      {label:`${key}-${L.id}`,phase,schema:SCHEMA}).then(r=>({paper:key,reviewer:L.id,area:L.a,...r})).catch(()=>null)))
}
const [a,b]=await Promise.all([panel('A',A_RUBRIC,A_lenses,'ReviewA-NLP'),panel('B',B_RUBRIC,B_lenses,'ReviewB-HCI')])
return {A:a.filter(Boolean), B:b.filter(Boolean)}
