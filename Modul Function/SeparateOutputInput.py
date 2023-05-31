# make function separate input, output
def SeparateOutputInput(data,
                       output_column_name):
    """
    The Function for separate data input and output
    input data for target/output and the drop or separate
    make new variable for input and output data
    """
    
    output_data = data[output_column_name]
    input_data = data.drop(output_column_name,
                           axis = 1)
    
    return input_data, output_data