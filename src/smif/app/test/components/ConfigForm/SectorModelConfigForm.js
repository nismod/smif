import React from 'react'
import {expect} from 'chai'
import {mount} from 'enzyme'
import {describe, it} from 'mocha'

import SectorModelConfigForm from '../../../src/components/ConfigForm/SectorModelConfigForm.js'
import ReactModal from 'react-modal'

import {sos_models, sector_model} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<SectorModelConfigForm />', () => {

    it('renders all info', () => {
        let wrapper = mount(<SectorModelConfigForm
            sosModels={sos_models}
            sectorModel={sector_model} />
        )

        const sector_model_name = wrapper.find('[id="sector_model_name"]')
        expect(sector_model_name.html()).to.contain(sector_model.name)

        const sector_model_description = wrapper.find('[id="sector_model_description"]')
        expect(sector_model_description.html()).to.contain(sector_model.description)

        const sector_model_classname = wrapper.find('[id="sector_model_classname"]')
        expect(sector_model_classname.html()).to.contain(sector_model.classname)

        const sector_model_path = wrapper.find('[id="sector_model_path"]')
        expect(sector_model_path.html()).to.contain(sector_model.path)

        const sector_model_input_0 = wrapper.find('[id="inputs_property_0"]')
        expect(sector_model_input_0.html()).to.contain(sector_model.inputs[0].name)
        expect(sector_model_input_0.html()).to.contain(sector_model.inputs[0].spatial_resolution)
        expect(sector_model_input_0.html()).to.contain(sector_model.inputs[0].temporal_resolution)
        expect(sector_model_input_0.html()).to.contain(sector_model.inputs[0].units)

        const sector_model_output_0 = wrapper.find('[id="outputs_property_0"]')
        expect(sector_model_output_0.html()).to.contain(sector_model.outputs[0].name)
        expect(sector_model_output_0.html()).to.contain(sector_model.outputs[0].spatial_resolution)
        expect(sector_model_output_0.html()).to.contain(sector_model.outputs[0].temporal_resolution)
        expect(sector_model_output_0.html()).to.contain(sector_model.outputs[0].units)

        const sector_model_parameter_0 = wrapper.find('[id="parameters_property_0"]')
        expect(sector_model_parameter_0.html()).to.contain(sector_model.parameters[0].name)
        expect(sector_model_parameter_0.html()).to.contain(sector_model.parameters[0].description)
        expect(sector_model_parameter_0.html()).to.contain(sector_model.parameters[0].default_value)
        expect(sector_model_parameter_0.html()).to.contain(sector_model.parameters[0].absolute_range)
        expect(sector_model_parameter_0.html()).to.contain(sector_model.parameters[0].suggested_range)
        expect(sector_model_parameter_0.html()).to.contain(sector_model.parameters[0].units)
    })
    
    it('renders warning when data incorrect', () => {
        let wrapper = mount(<SectorModelConfigForm
            sosModels={empty_array}
            sectorModel={empty_object} />
        )

        expect(wrapper.html()).to.contain('This Scenario Set does not exist')
    })

    it('add and delete input', () => {
        let wrapper = mount(<SectorModelConfigForm
            sosModels={sos_models}
            sectorModel={sector_model} />
        )

        // Open the Add input popup
        wrapper.find('input#btn_add_input').simulate('click')

        // Check if the form opens
        let popup_add_input = wrapper.find('[id="popup_add_input"]')
        expect(popup_add_input.exists()).to.equal(true)
        
        // Fill in form
        wrapper.find('input#input_name').simulate('change', { target: { name: 'name', value: 'test_name'} })
        wrapper.find('input#input_units').simulate('change', { target: { name: 'units', value: 'test_units'} })
        wrapper.find('input#input_spatial_res').simulate('change', { target: { name: 'spatial_resolution', value: 'test_spatial_resolution'} })
        wrapper.find('input#input_temporal_res').simulate('change', { target: { name: 'temporal_resolution', value: 'test_temporal_resolution'} })

        // Submit form
        wrapper.find('input#btn_input_save').simulate('submit')

        // Check if inputs were added
        expect(wrapper.state().selectedSectorModel.inputs[2].name).to.equal('test_name')
        expect(wrapper.state().selectedSectorModel.inputs[2].units).to.equal('test_units')
        expect(wrapper.state().selectedSectorModel.inputs[2].spatial_resolution).to.equal('test_spatial_resolution')
        expect(wrapper.state().selectedSectorModel.inputs[2].temporal_resolution).to.equal('test_temporal_resolution')

        // delete the entry that was added
        wrapper.find('[id="inputs_property_2"]').find('button').simulate('click')

        // check if the popup shows the name of the input (test_name)
        expect(wrapper.find('[id="popup_delete_form"]').html()).to.contain('test_name')

        // confirm delete
        wrapper.find('[id="popup_delete_form"]').find('input#deleteButton').simulate('click')

        // confirm input was deleted
        expect(wrapper.state().selectedSectorModel.inputs[2]).to.equal(undefined)
    })
})
