import { combineReducers } from 'redux';

export const configurations = (state = {}, action) => {
    switch (action.type) {
        case 'FETCH_PROJECTS':
            return {
                ...state,
                projects:["Project1", "Project2", "Project3"],
                error: null,
                loading: true,
                hello_world: "Hello World"
            };
        default:
            return state;
    }
};

export const reducers = combineReducers({
    configurations
});
