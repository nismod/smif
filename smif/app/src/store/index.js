import { applyMiddleware, combineReducers, createStore} from 'redux';
import { reducers } from '../reducers'

export function configureStore(initialState = {}) {
    const store = createStore(
      	reducers,
		initialState,
		window.__REDUX_DEVTOOLS_EXTENSION__ && window.__REDUX_DEVTOOLS_EXTENSION__(),
    )
    return store;
};
  
export const store = configureStore();