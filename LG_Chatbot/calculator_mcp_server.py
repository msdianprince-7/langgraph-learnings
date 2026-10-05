from mcp.server.fastmcp import FastMCP

# An MCP server is a separate program that offers tools. Any MCP client (our chatbot,
# Claude Desktop, an IDE...) can start it and call its tools.
mcp = FastMCP('calculator')


@mcp.tool()
def calculator(first_num: float, second_num: float, operation: str) -> str:
    """Do basic arithmetic on two numbers. operation is one of: add, sub, mul, div."""
    if operation == 'add':
        return str(first_num + second_num)
    elif operation == 'sub':
        return str(first_num - second_num)
    elif operation == 'mul':
        return str(first_num * second_num)
    elif operation == 'div':
        return str(first_num / second_num) if second_num != 0 else 'Division by zero is not allowed'
    return f'Unsupported operation: {operation}'


if __name__ == '__main__':
    # stdio: the client starts this file as a subprocess and talks to it over stdin/stdout
    mcp.run(transport='stdio')
