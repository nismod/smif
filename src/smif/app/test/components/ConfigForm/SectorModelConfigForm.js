import React from 'react'
import { expect } from 'chai'
import { shallow } from 'enzyme'
import SectorModelConfigForm from '../../../src/components/ConfigForm/SectorModelConfigForm.js'

import {sector_model, sector_models, scenario_sets, narrative_sets} from '../../helpers.js'
import {empty_object, empty_array} from '../../helpers.js'

describe('<SectorModelConfigForm />', () => {

    const correctRender = shallow(<SectorModelConfigForm sectorModel={sector_model} />)
    const dataMissingRender = shallow(<SectorModelConfigForm sectorModel={empty_object} />)

    it('renders sector_model.name', () => {
        const sector_model_name = correctRender.find('[id="sector_model_name"]')
        expect(sector_model_name.html()).to.contain(sector_model.name)
    })

    it('renders sector_model.name when data missing', () => {
        const sector_model_name = dataMissingRender.find('[id="sector_model_name"]')
        expect(sector_model_name.html()).to.contain(`id="sector_model_name"`)
    })

    it('renders sector_model.description', () => {
        const sector_model_description = correctRender.find('[id="sector_model_description"]')
        expect(sector_model_description.html()).to.contain(sector_model.description)
    })

    it('renders sector_model.description when data missing', () => {
        const sector_model_description = dataMissingRender.find('[id="sector_model_description"]')
        expect(sector_model_description.html()).to.contain(`id="sector_model_description"`)
    })
})
