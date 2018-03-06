import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import PropertyList from '../../../../src/components/ConfigForm/General/PropertyList.js'


import {sos_model} from '../../../helpers.js'
import {empty_object, empty_array} from '../../../helpers.js'

describe('<PropertyList />', () => {

    it('renders properties', () => {
        const wrapper = shallow(<PropertyList itemsName="dependencies" items={sos_model.dependencies} columns={{source_model: 'Source', source_model_output: 'Output', sink_model: 'Sink', sink_model_input: 'Input'}} />)

        for (let i = 0; i < sos_model.dependencies.length; i++) {
            let expectedRow = '<tr id="property_' + i + '">'
            let columns = ['source_model', 'source_model_output', 'sink_model', 'sink_model_input']

            columns.forEach(function(key) {
                expectedRow += '<td width="25%">' + sos_model.dependencies[i][key] + '</td>'
            })
            expect(wrapper.find('[id="property_'+ i +'"]').html()).to.contain(expectedRow)
        }
    })

    it('renders properties with editButton', () => {
        const wrapper = shallow(<PropertyList itemsName="dependencies" items={sos_model.dependencies} columns={{source_model: 'Source', source_model_output: 'Output', sink_model: 'Sink', sink_model_input: 'Input'}} editButton={true} />)

        for (let i = 0; i < sos_model.dependencies.length; i++) {
            let expectedRow = '<tr id="property_' + i + '">'
            let columns = ['source_model', 'source_model_output', 'sink_model', 'sink_model_input']

            columns.forEach(function(key) {
                expectedRow += '<td width="23%">' + sos_model.dependencies[i][key] + '</td>'
            })
            expect(wrapper.find('[id="property_'+ i +'"]').html()).to.contain(expectedRow)
            expect(wrapper.find('[id="property_'+ i +'"]').html()).to.contain('<td width="8%"><button type="button" class="btn btn-outline-dark" name="edit">')
        }
    })

    it('renders properties with deleteButton', () => {
        const wrapper = shallow(<PropertyList itemsName="dependencies" items={sos_model.dependencies} columns={{source_model: 'Source', source_model_output: 'Output', sink_model: 'Sink', sink_model_input: 'Input'}} deleteButton={true} />)

        for (let i = 0; i < sos_model.dependencies.length; i++) {
            let expectedRow = '<tr id="property_' + i + '">'
            let columns = ['source_model', 'source_model_output', 'sink_model', 'sink_model_input']

            columns.forEach(function(key) {
                expectedRow += '<td width="23%">' + sos_model.dependencies[i][key] + '</td>'
            })
            expect(wrapper.find('[id="property_'+ i +'"]').html()).to.contain(expectedRow)
            expect(wrapper.find('[id="property_'+ i +'"]').html()).to.contain('<button type="button" class="btn btn-outline-dark" value="dependencies">')
        }
    })

    it('renders properties with warnings', () => {
        const wrapper = shallow(<PropertyList itemsName="dependencies" items={sos_model.dependencies} columns={{source_model: 'Source', source_model_output: 'Output', sink_model: 'Sink', sink_model_input: 'Input'}} enableWarnings={true} rowWarning={[true, true, true, true, true]} />)

        for (let i = 0; i < sos_model.dependencies.length; i++) {
            let expectedRow = ''
            let columns = ['source_model', 'source_model_output', 'sink_model', 'sink_model_input']

            columns.forEach(function(key) {
                expectedRow += '<td width="23%">' + sos_model.dependencies[i][key] + '</td>'
            })
            expect(wrapper.find('[id="property_'+ i +'"]').html()).to.contain(expectedRow)
            expect(wrapper.find('[id="property_'+ i +'"]').html()).to.contain('warning')
        }
    })

    it('renders properties with warnings enabled but not activated', () => {
        const wrapper = shallow(<PropertyList itemsName="dependencies" items={sos_model.dependencies} columns={{source_model: 'Source', source_model_output: 'Output', sink_model: 'Sink', sink_model_input: 'Input'}} enableWarnings={true} rowWarning={[false, false, false, false, false]} />)

        for (let i = 0; i < sos_model.dependencies.length; i++) {
            let expectedRow = ''
            let columns = ['source_model', 'source_model_output', 'sink_model', 'sink_model_input']

            columns.forEach(function(key) {
                expectedRow += '<td width="23%">' + sos_model.dependencies[i][key] + '</td>'
            })
            expect(wrapper.find('[id="property_'+ i +'"]').html()).to.contain(expectedRow)
            expect(wrapper.find('[id="property_'+ i +'"]').html()).to.not.contain('warning')
        }
    })
})
