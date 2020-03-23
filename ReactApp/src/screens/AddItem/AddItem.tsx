import React from "react"
import {
    KeyboardAvoidingView,
    Text,
    TextInput,
    View,
    TouchableOpacity,
    ScrollView,
    Image,
} from "react-native";
import Constants from "expo-constants";
import axios from "axios"

import {LinearGradient} from 'expo-linear-gradient'

// @ts-ignore
function FormField({title, placeholder, onChange, index}: TFormFields) {
    return (
        <View style={{marginBottom: 5}}>
            <View>
                <Text style={{color: "#303f9f"}}>
                    {title}
                </Text>
            </View>
            <TextInput
                {...{placeholder}}
                blurOnSubmit={true}
                clearTextOnFocus={true}
                style={[{
                    borderBottomWidth: 1,
                    marginBottom: 7.5,
                    borderBottomColor: "#303f9f"
                }]}
                onChangeText={text => onChange({text, index})}
            />
        </View>
    )
}


const itemFields = [
    {
        title: "Item Name",
        placeholder: "Moong Ki Dal"
    },
    {
        title: "MRP",
        placeholder: "120"
    },
    {
        title: "Manufacturing Date",
        placeholder: "2050-10-21"
    },
    {
        title: "Expiry Date",
        placeholder: "2050-12-21"
    },
    {
        title: "Quantity",
        placeholder: "1 kg/1 lt/ 1 Packet"
    },
];

let temp = {
    product_name: "",
    mrp: 0,
    mfg_date: '',
    exp_date: '',
    quantity: ""
};

export default function AddItem(props) {

    const [itemState, setItemState] = React.useState({})

    function onChange({text, index}) {


        switch (index) {
            case 0:
                temp = {
                    ...temp,
                    product_name: text
                };
                break;
            case 1: {
                temp = {
                    ...temp,
                    mrp: parseInt(text)
                }
                break;
            }
            case 2: {
                temp = {
                    ...temp,
                    mfg_date: text
                }
                break;
            }
            case 3: {
                temp = {
                    ...temp,
                    exp_date: text
                }
                break;
            }
            case 4: {
                temp = {
                    ...temp,
                    quantity: text
                }
                break;
            }
            default: {
                break;
            }
        }

        setItemState(temp)

    }

    function submitdata() {
        // @ts-ignore
        itemState.mrp = parseInt(itemState.mrp)

        axios.post("https://foodkart-abf29.firebaseio.com/item.json", [itemState])
            .then(res => props.navigation.navigate("Items", {reload: true}))
            .catch(res => console.log({res}))

    }


    return (
        <KeyboardAvoidingView style={[{flex: 1, marginTop: Constants.statusBarHeight}]} behavior="padding" enabled
                              contentContainerStyle={{flexGrow: 1}}>
            <View style={{flex: 1, justifyContent: "center", alignItems: "center"}}>
                <Image
                    source={{uri: "https://cdn.dribbble.com/users/1111912/screenshots/4391205/3.png"}}
                    style={{
                        flex: 1,
                        width: "100%",
                        height: "100%",
                        resizeMode: "contain"
                    }}
                />
            </View>
            <ScrollView
                contentContainerStyle={{
                    justifyContent: "center",
                    borderTopLeftRadius: 20,
                    borderTopRightRadius: 20,
                    padding: 30,
                    backgroundColor: "white"
                }}
                style={{
                    flex: 1,
                }}>
                {
                    // @ts-ignore
                    itemFields.map(({title, placeholder}, key) => (
                        <FormField {...{title, placeholder}} index={key} onChange={onChange} key={key}/>
                    ))
                }
                <TouchableOpacity onPress={submitdata}>
                    <Text style={{textAlign: "center"}}>
                        Submit
                    </Text>
                </TouchableOpacity>
            </ScrollView>
        </KeyboardAvoidingView>
    )
}
