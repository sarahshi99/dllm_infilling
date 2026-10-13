// DOM API harness for the exact embedded browser script; does not run candidates.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const out=path.join(__dirname,'../analysis_outputs/dreamon_markov_error_audit_20261013_v1');
const doc=fs.readFileSync(path.join(out,'casebook.html'),'utf8');
const scripts=[...doc.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
class Element{
 constructor(tag){this.tag=tag;this.children=[];this.events={};this.dataset={};this._text='';this._value='';this.className='';}
 set textContent(x){this._text=String(x);this.children=[]}get textContent(){return this._text+this.children.map(c=>c.textContent).join('')}
 set value(x){this._value=x}get value(){return this.tag==='option'?this._value||this._text:this._value}
 append(...items){this.children.push(...items)}replaceChildren(...items){this._text='';this.children=items}
 addEventListener(name,fn){this.events[name]=fn}get firstElementChild(){return this.children[0]}
}
const ids=[...doc.matchAll(/\bid="([^"]+)"/g)].map(x=>x[1]);const nodes=Object.fromEntries(ids.map(id=>[id,new Element('div')]));
nodes['audit-data'].textContent=scripts[0][1];nodes.scope.value='failures';
const context=vm.createContext({document:{getElementById:id=>nodes[id],createElement:t=>new Element(t),createTextNode:t=>new Element('text')},location:{hash:''},window:{addEventListener(){}},console});
context.document.createTextNode=t=>{const e=new Element('text');e.textContent=t;return e};
vm.runInContext(scripts[1][1],context,{timeout:10000});
const run=s=>vm.runInContext(s,context,{timeout:10000});
assert.equal(run('filtered.length'),326);assert.equal(nodes.cases.children.length,25);
run("$('policy').value='global_confidence';filter()");assert.equal(run('filtered.length'),144);
run("$('policy').value='left_to_right_frontier';filter()");assert.equal(run('filtered.length'),182);
run("$('reset').onclick();$('scope').value='all';filter()");assert.equal(run('filtered.length'),2066);
run("$('status').value='无需失败分析';filter()");assert.equal(run('filtered.length'),1740);
run("$('reset').onclick();$('bits').value='11101';filter()");assert(run('filtered.length')>0);assert(run("filtered.every(c=>c.bits==='11101')"));
run("$('reset').onclick();$('abd').value='000';filter()");assert(run("filtered.every(c=>c.abd==='000')"));
run("$('reset').onclick();$('error').value='语法错误';filter()");assert(run("filtered.every(c=>c.error_types.includes('语法错误'))"));
run("$('reset').onclick();$('level').value='待验证假设';filter()");assert.equal(run('filtered.length'),76);
run("$('reset').onclick();$('base').value='101';filter()");assert(run("filtered.every(c=>c.base_task_id==='101')"));
run("$('reset').onclick();$('search').value='return return';filter()");assert(run('filtered.length')>0);
run("$('reset').onclick();$('next').onclick()");assert.equal(run('page'),1);run("$('prev').onclick()");assert.equal(run('page'),0);
context.location.hash='#'+encodeURIComponent('global_confidence::SingleLineInfilling/HumanEval/1/L4');run('navigate()');assert.equal(run('filtered.length'),1);
const d=nodes.cases.firstElementChild;d.events.toggle();assert(d.textContent.includes('四组把 if c == 条件丢掉'));
assert(d.textContent.includes("' ':"));assert(d.textContent.includes('原prefix'));
run("const adversarial=pre('</script><img src=x onerror=alert(1)>\\n  中文 \"引号\"');if(adversarial.children.length!==0)throw Error('unsafe DOM')");
run("diffView(document.createElement('div'),'  x=1\\n','  x=2\\n')");
fs.writeFileSync(path.join(out,'html_verification.json'),JSON.stringify({status:'passed',method:'Node executes exact page JavaScript with minimal DOM API harness',limitations:'No installed browser; visual layout/native browser rendering not verified',checks:['default326','both policies','all2066','all-pass1740','five-bit and eight-class filters','error/level/base filters','code search','pagination','stable-ID navigation','expanded Chinese/quotes/newlines','inert adversarial text','switchable final text diff'],utc:new Date().toISOString()},null,2)+'\n');
console.log('HTML DOM/JavaScript verification passed; native browser rendering unavailable.');
