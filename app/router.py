import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter

encoder = HuggingFaceEncoder(
    name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

faq = Route(
    name='faq',
    utterances=[
        "What is the return policy of the products?",
        "Do I get discount with the HDFC credit card?",
        "How can I track my order?",
        "What payment methods are accepted?",
        "How long does it take to process a refund?",
        "return policy",
        "product return",
        "refund policy",
        "refund processing time",
        "track my order",
        "order tracking",
        "order status",
        "cancel an order",
        "modify an order",
        "payment methods",
        "accepted payment options",
        "cash on delivery",
        "online payment options",
        "credit card discount",
        "HDFC card offer",
        "bank discount",
        "sales and promotions",
        "ongoing offers",
        "defective products",
        "defective product policy",
        "policy for defective products",
        "return policy for defective items",
        "refund for defective product",
        "damaged or defective product",
        "faulty product return policy",
        "is my amount refundable?",
    ]
)

sql = Route(
    name='sql',
    utterances=[
        "I want to buy saree that is below 3k.",
        "shoes under 3k",
        "watch under 5k",
        "saree under 3k",
        "laptop under 50k",
        "earbuds under 2k",
        "men shirt under 1k",
    ]
)

general_qa = Route(
    name='general_qa',
    utterances=[
        "who are you?",
        "Who are you?",
        "what is your name?",
        "What can you help me with?",
        "Are you a real person?",
        "How do you work?",
        "Can you help me choose a product?",
        "What are the best products right now?",
        "Can you show products within my budget?",
        "Are there any good deals available?",
        "Is this product worth buying?",
        "Can you compare two products?",
        "Is there a cheaper alternative?",
        "Can you recommend products for daily use?",
        "What can’t you do?",
        "Can you help me shop faster?",
        "Thanks",
        "Thank you",
        "Who are you exactly?",
        "What are you?",
        "Can you tell me who you are?",
        "What kind of assistant are you?",
        "Are you a bot?",
        "Tell me about yourself",
        "What do you do?",
        "What should I call you?",
        "Do you have a name?",
        "May I know your name?",
        "What’s your name?",
        "How can I address you?",
        "What can you do for me?",
        "How can you help?",
        "What kind of help do you provide?",
        "What services do you offer?",
        "What are you capable of?",
        "How can you assist me?",
        "Are you human?",
        "Are you a chatbot?",
        "Am I talking to a bot?",
        "Is this a real person?",
        "Are you AI?",
        "How does this work?",
        "How do you find products?",
        "How do you give recommendations?",
        "What’s your working process?",
        "How do you understand my queries?",
        "Hi",
        "Hello",
        "what are the queries that I made above"
    ]
)



router = SemanticRouter(
    routes=[faq, sql, general_qa],
    encoder=encoder,
    auto_sync="local",
)


if __name__ == "__main__":
    print(router("What is your policy on defective product").name)
    print(router("shoes in price range 5000 to 1000").name)
    print(router("what is your role").name)
