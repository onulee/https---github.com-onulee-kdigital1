import logo from './logo.svg';
import './App.css';
import React, { useState } from 'react';
import Counter from './components/Counter';

function App() {
  const [buttonName, setButtonName] = useState('버튼');
  const clickButton = () => {
    setButtonName('클릭');
  };
  return (
    <div className="App">
      <h1>main 페이지</h1>
      <Counter click="click1" />
      <Counter click={buttonName} />
      <Counter />
      <p />
      <button onClick={clickButton}>이름변경</button>
    </div>
  );
}

export default App;