import React, { Component } from 'react';
import { connect } from 'react-redux';
import { createSelector } from 'reselect';
import { Button } from 'semantic-ui-react';
import InfoTabStyles from '../styles/InfoTabStyles';
import { worldRegions } from '../../assets/mapViewSettings';
import { openRegionModal } from '../../actions/mapActions';

class InfoTab extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isFlagImgPresent: true,
      isFlagImgReady: false,
    };
  }

  componentDidMount() {
    const { flag } = this.props.selectedProperties;
    const flagImg = new Image();
    flagImg.src = flag;
    if (!flag) {
      this.setState({ isFlagImgPresent: false });
      return;
    }

    flagImg.onload = () => {
      this.setState({ isFlagImgReady: true });
    };
  }

  render() {
    const {
      selectedProperties,
      currentMap,
      isQuizActive,
      openRegionModal,
    } = this.props;
    let {
      name,
      capital,
      population,
      area,
      regionOf,
      flag,
    } = selectedProperties;
    const { isFlagImgPresent, isFlagImgReady } = this.state;
    if (!isFlagImgReady && isFlagImgPresent) return '';
    population = population ? `${population.toLocaleString()}` : 'N/A';
    area = area ? `${area.toLocaleString()} km²` : 'N/A';
    const showMoreInfoButton =
      !isQuizActive && worldRegions.includes(currentMap);
    return (
      <InfoTabStyles>
        {isFlagImgPresent && (
          <img className="infoTab-flag" src={flag} alt={`${name}-flag`} />
        )}
        <div className="infoTab-desc">
          <li>Pais: {name}</li>
          <li>Capital: {capital}</li>
          <li>Población: {population}</li>
          <li>Area: {area}</li>
          {regionOf ? <li>Region of {regionOf}</li> : ''}
          
        </div>
      </InfoTabStyles>
    );
  }
}

const getAppState = createSelector(
  state => state.quiz.selectedProperties,
  state => state.quiz.isQuizActive,
  state => state.map.currentMap,
  (selectedProperties, isQuizActive, currentMap) => ({
    selectedProperties,
    isQuizActive,
    currentMap,
  })
);

export default connect(getAppState, { openRegionModal })(InfoTab);
