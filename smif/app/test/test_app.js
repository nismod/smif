import React from 'react';
import { expect } from 'chai';
import { shallow } from 'enzyme';
import App from '../src/hello/app.jsx';

describe('<App />', () => {
    it('renders heading text', () => {
        const wrapper = shallow(<App heading='test' />);
        expect(wrapper.find('h1').text()).to.equal('test');
    });
});
