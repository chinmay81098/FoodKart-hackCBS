import React from "react"
import {ImageBackground, Text, TouchableNativeFeedback, TouchableOpacity, View} from 'react-native'
import {withNavigation} from "react-navigation";

// export default class GetStarted extends React.PureComponent {
//     render() {
//
//     }
// }

function GetStartedComponent(props) {
    return (
        <ImageBackground
            source={{uri: "https://cdn.dribbble.com/users/1111912/screenshots/4391205/3.png"}}
            style={{
                flex: 1,
                width: "100%",
                height: "100%",
            }}
            imageStyle={{
                resizeMode: "contain" // works only here!
            }}
        >
            <View
                style={{
                    flex: 1,
                    justifyContent: "flex-end",
                    alignItems: "center",
                    marginBottom: 20
                }}
            >
                <TouchableOpacity
                    onPress={() => props.navigation.navigate("AuthScreen")}
                    style={{
                        padding: 10,
                        borderRadius: 10,
                        borderWidth: 1,
                        borderColor: "#0A10C8"
                    }}>
                    <Text style={{color: "#0A10C8"}}>
                        Get Started
                    </Text>
                </TouchableOpacity>
            </View>
        </ImageBackground>
    );
}


export default withNavigation(GetStartedComponent)
