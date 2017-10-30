import { reducers } from '../reducers'

export const fetchProjects = () => ({
  type: 'FETCH_PROJECTS'
});

export const fetchProjectsSuccess = () => ({
  type: 'FETCH_PROJECTS_SUCCESS'
});

export const fetchProjectsFailure = () => ({
  type: 'FETCH_PROJECTS_FAILURE'
});