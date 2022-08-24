import logo from './logo.svg';
import './App.css';

function App() {

  let onSubmit = ()=>{
    alert('데이터를 전송합니다')
  }

  let onKeyUp = (event)=>{
    console.log('input 글자입력')
    if (event.keyCode == 13){
      onSubmit()
    }
  }

  return (
    <div className="App">
      <input onKeyUp={onKeyUp}></input>
      <button onClick={onSubmit}>버튼클릭</button>
    </div>

  );
}

export default App;
