#set selected database
selected_database = 'local_database'
# Operator mapping for conditions
operator_mapping = {
    "greater than or equal to": ">=",
    "less than or equal to": "<=",
    "greater than": ">",
    "less than": "<",
    "equal to": "=",
    "equals": "=",
    "not equal to": "<>",
    "not equal": "<>",
    "is": "==",
    "is not": "<>",
    "isn't": "<>"
}

# Aggregation function mapping
aggregation_mapping = {
    "average": "AVG",
    "maximum": "MAX",
    "minimum": "MIN",
    "sum": "SUM",
    "total": "SUM",
    "count": "COUNT"
}

def tokenize_nl_query(nl_query):
    # Remove punctuation and split into words
    nl_query = re.sub(r'[^\w\s]', '', nl_query)
    tokens = nl_query.lower().split()
    return tokens

def format_value(value):
    try:
        float(value)
        return value  # It's a number
    except ValueError:
        return f"'{value}'"  # It's a string

def parse_condition(condition_tokens, aggregation_function=None, aggregation_field=None):
    condition_str = ' '.join(condition_tokens)

    # Replace natural language operators with SQL operators
    for nl_op in sorted(operator_mapping.keys(), key=lambda x: -len(x)):
        if nl_op in condition_str:
            sql_op = operator_mapping[nl_op]
            condition_str = condition_str.replace(nl_op, sql_op)
            break  # Assume only one operator per condition

    # Replace aggregation function name with SQL function if applicable
    if aggregation_function and aggregation_function.lower() in condition_str:
        condition_str = condition_str.replace(aggregation_function.lower(), f"{aggregation_function}({aggregation_field})")

    # Split the condition into field, operator, and value
    match = re.match(r'([\w()]+)\s*([><=!]+)\s*(.+)', condition_str)
    if match:
        field = match.group(1)
        operator = match.group(2)z
        value = format_value(match.group(3).strip())
        return f"{field} {operator} {value}"
    else:
        return condition_str  # Return as is if parsing fails

def find_common_columns(table1, table2, database_info):
    columns1 = set(database_info[selected_database][table1]["columns"].keys())
    columns2 = set(database_info[selected_database][table2]["columns"].keys())
    common_columns = columns1.intersection(columns2)
    return list(common_columns)

def parse_nl_query(nl_query, database_info):
    tokens = tokenize_nl_query(nl_query)
    token_index = 0

    # Step 1: Check if first word is 'Calculate' or not
    is_aggregation = False
    if tokens[token_index] == 'calculate':
        is_aggregation = True
        token_index += 1
    elif tokens[token_index] in ['show', 'find']:
        token_index += 1
    else:
        return "Error: Query should start with 'Show', 'Find', or 'Calculate'."

    # Step 2: Check if next word is a number (Limit clause)
    limit_number = None
    if token_index < len(tokens) and tokens[token_index].isdigit():
        limit_number = tokens[token_index]
        token_index += 1

    # Step 3: Get all fields or aggregation of field
    fields = []
    aggregation_function = None
    if is_aggregation:
        # Expecting an aggregation function
        if token_index < len(tokens):
            agg_word = tokens[token_index]
            if agg_word in aggregation_mapping:
                aggregation_function = aggregation_mapping[agg_word]
                token_index += 1
            else:
                return f"Error: Unknown aggregation function '{agg_word}'."

            # Expecting 'of'
            if token_index < len(tokens) and tokens[token_index] == 'of':
                token_index += 1
            else:
                return "Error: Expected 'of' after aggregation function."

            # Get the field for aggregation
            if token_index < len(tokens):
                field = tokens[token_index]
                fields.append(field)
                token_index += 1
            else:
                return "Error: Expected field after 'of'."
        else:
            return "Error: Incomplete aggregation function."
    else:
        # Non-aggregation: Get fields
        while token_index < len(tokens) and tokens[token_index] not in ['in', 'of']:
            field = tokens[token_index].strip(',')
            fields.append(field)
            token_index += 1

    # Step 4: Get all the tables
    if token_index < len(tokens) and tokens[token_index] in ['in', 'of']:
        token_index += 1
    else:
        return "Error: Expected 'in' or 'of' before table names."

    tables = []
    while token_index < len(tokens) and tokens[token_index] not in ['by', 'with', 'sorted', 'order']:
        table = tokens[token_index].strip(',')
        tables.append(table)
        token_index += 1

    if len(tables) == 0:
        return "Error: No table specified."

    # Handle JOIN if two tables
    join_condition = None
    if len(tables) == 2:
        common_columns = find_common_columns(tables[0], tables[1], database_info)
        if common_columns:
            # Use the first common column
            common_column = common_columns[0]
            join_condition = f"{tables[0]}.{common_column} = {tables[1]}.{common_column}"
        else:
            return "Error: No common columns found for JOIN between tables."

    # Step 5: Check for 'by' (GROUP BY)
    group_by_field = None
    if token_index < len(tokens) and tokens[token_index] == 'by':
        token_index += 1
        if token_index < len(tokens):
            group_by_field = tokens[token_index]
            token_index += 1
        else:
            return "Error: Expected field after 'by'."

    # Step 6: Check for 'with' (Condition)
    condition = None
    if token_index < len(tokens) and tokens[token_index] == 'with':
        token_index += 1
        condition_tokens = []
        while token_index < len(tokens) and tokens[token_index] not in ['sorted', 'order']:
            condition_tokens.append(tokens[token_index])
            token_index += 1
        condition = parse_condition(condition_tokens, aggregation_function=aggregation_function, aggregation_field=fields[0])

    # Step 7: Check for 'sorted by' or 'order by' (ORDER BY)
    order_by_field = None
    if token_index < len(tokens):
        if tokens[token_index] in ['sorted', 'order']:
            token_index += 1
            if token_index < len(tokens) and tokens[token_index] == 'by':
                token_index += 1
                if token_index < len(tokens):
                    order_by_field = tokens[token_index]
                    token_index += 1
                else:
                    return "Error: Expected field after 'by'."
            else:
                return "Error: Expected 'by' after 'sorted' or 'order'."

    # Construct the SQL query
    sql_query_parts = []

    # SELECT clause
    select_clause = "SELECT "
    if is_aggregation:
        # Include an alias for the aggregated field
        select_clause += f"{aggregation_function}({fields[0]}) AS {aggregation_function.lower()}"
    else:
        if len(table) > 1: 
#             if join, add alias to the fields
            for f in fields:
                if f in database_info[selected_database][tables[0]]["columns"]:
                    select_clause += f"{tables[0]}.{f}, "
                elif f in database_info[selected_database][tables[1]]["columns"]:
                    select_clause += f"{tables[1]}.{f}, "
        else:
            select_clause += ', '.join(fields)

    # FROM clause
    from_clause = "FROM "
    if len(tables) == 1:
        from_clause += tables[0]
    elif len(tables) == 2:
        from_clause += f"{tables[0]} JOIN {tables[1]} ON {join_condition}"
    else:
        return "Error: More than two tables are not supported."

    sql_query_parts.append(select_clause)
    sql_query_parts.append(from_clause)

    # WHERE clause (for non-aggregation or pre-aggregation filtering)
    where_clause = None
    having_clause = None

    if condition:
        if is_aggregation and group_by_field:
            # Use HAVING clause
            having_clause = f"HAVING {condition}"
        else:
            # Use WHERE clause
            where_clause = f"WHERE {condition}"

    if where_clause:
        sql_query_parts.append(where_clause)

    # GROUP BY clause
    if group_by_field:
        sql_query_parts.append(f"GROUP BY {group_by_field}")

    # HAVING clause
    if having_clause:
        sql_query_parts.append(having_clause)

    # ORDER BY clause
    if order_by_field:
        if is_aggregation and order_by_field == aggregation_function.lower():
            # ORDER BY aggregation function
            order_by_clause = f"ORDER BY {aggregation_function}({fields[0]})"
        else:
            order_by_clause = f"ORDER BY {order_by_field}"
        sql_query_parts.append(order_by_clause)

    # LIMIT clause
    if limit_number:
        sql_query_parts.append(f"LIMIT {limit_number}")

    # Combine all parts
    sql_query = '\n'.join(sql_query_parts) + ';'

    return sql_query