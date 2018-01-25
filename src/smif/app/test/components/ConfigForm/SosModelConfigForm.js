import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import SosModelConfigForm from '../../../src/components/ConfigForm/SosModelConfigForm.js'

import {sos_model, sector_models, scenario_sets, narrative_sets} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<SosModelConfigForm />', () => {

    const correctRender = shallow(<SosModelConfigForm sosModel={sos_model} sectorModels={sector_models} scenarioSets={scenario_sets} narrativeSets={narrative_sets} />)
    const dataMissingRender = shallow(<SosModelConfigForm sosModel={empty_object} sectorModels={empty_array} scenarioSets={empty_array} narrativeSets={empty_array} />)

    it('renders sos_model.name', () => {
        const sos_model_name = correctRender.find('[id="sos_model_name"]')
        expect(sos_model_name.html()).to.contain(sos_model.name)
    })

    it('renders sos_model.name when data missing', () => {
        const sos_model_name = dataMissingRender.find('[id="sos_model_name"]')
        expect(sos_model_name.html()).to.contain(`id="sos_model_name"`)
    })

    it('renders sos_model.description', () => {
        const sos_model_description = correctRender.find('[id="sos_model_description"]')
        expect(sos_model_description.html()).to.contain(sos_model.description)
    })

    it('renders sos_model.description when data missing', () => {
        const sos_model_description = dataMissingRender.find('[id="sos_model_description"]')
        expect(sos_model_description.html()).to.contain(`id="sos_model_description"`)
    })
})
