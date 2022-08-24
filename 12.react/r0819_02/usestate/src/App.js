import logo from './logo.svg';
import './App.css';
import React, {useState} from 'react';

function App() {

  const [text,setText] = useState('hello')
  let txt='hello'
  
  let changeTxt = () =>{
    setText('changeHello') // 글자는 changeHello변경
    console.log(txt)   // hello출력
  }
  let onSubmit = ()=>{
    alert('데이터를 전송합니다')
  }

  let onKeyUp = (event)=>{
    console.log('input 글자입력 됨')
    if (event.keyCode == 13){ //enter키
      onSubmit()
    }
  }

  return (
    <div className="App">
      <input onKeyUp={onKeyUp}></input>
      <button onClick={onSubmit}>버튼클릭</button>
      <br />
      <br></br>
      <span>{text}</span>
      <button onClick={changeTxt} >글자변경</button>
    </div>

  );
}

export default App;
