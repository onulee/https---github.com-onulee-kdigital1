import logo from './logo.svg';
import './App.css';
import React, { useState } from 'react';

function App() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const onSubmit = (event) => {
    event.preventDefault(); // form은 페이지를 리셋함. 리셋방지
    console.log(username, password);
  };

  return (
    <div className="App">
      {/* submit실행시  */}
      <form onSubmit={onSubmit}> 
        <input 
          type="text"
          placeholder="Username" 
          value={username} 
          onChange={(e) => setUsername(e.target.value)}
        /><br />
        <input 
          type="password"
          placeholder="Password" 
          value={password} 
          onChange={(e) => setPassword(e.target.value)}
        /><br />
        <button type="submit">Login</button>
      </form>
    </div>
  );
}

export default App;
