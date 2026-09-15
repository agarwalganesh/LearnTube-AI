"""
Seed script to add a realistic educational video with notes and flashcards
so the UI can be explored immediately without consuming live API tokens.
"""
from app import create_app
from models.database import db, Video, Notes, Flashcard, ChatMessage

def seed():
    app = create_app()
    with app.app_context():
        # Check if already seeded
        existing = Video.query.filter_by(youtube_video_id="aircAruvnKk").first()
        if existing:
            print(f"Sample video already exists with ID: {existing.id}")
            return existing.id

        video = Video(
            youtube_url="https://www.youtube.com/watch?v=aircAruvnKk",
            youtube_video_id="aircAruvnKk",
            title="3Blue1Brown: But what is a neural network? | Deep learning, chapter 1",
            transcript=(
                "What is a neural network? Deep learning has achieved incredible breakthroughs in computer vision, "
                "natural language processing, and robotics. At its core, a neural network is a mathematical function "
                "inspired by biological neurons in the human brain. It consists of layers: an input layer, one or more "
                "hidden layers, and an output layer. In an image classification task, like recognizing handwritten digits "
                "from the MNIST dataset, each input neuron represents the grayscale brightness of a single pixel. "
                "The activation of a neuron is a number between 0 and 1. To compute the next layer's activations, we take "
                "the weighted sum of the inputs: a = sigma(w1*x1 + w2*x2 + ... + wn*xn + b), where w are weights, b is the bias, "
                "and sigma is an activation function such as the Sigmoid or ReLU. The weights determine what patterns each neuron "
                "responds to, like edges, loops, or diagonal strokes. During training, the network uses gradient descent and "
                "backpropagation to adjust the weights and biases to minimize a loss function, measuring the difference between "
                "predicted output and the true label."
            ),
            course_name="Deep Learning",
            completed=False
        )
        db.session.add(video)
        db.session.commit()

        notes = Notes(
            video_id=video.id,
            summary=(
                "An intuitive and visual introduction to artificial neural networks, explaining how layered "
                "mathematical transformations convert raw sensory inputs (such as image pixels) into abstract "
                "representations and classifications through weighted sums, biases, and activation functions."
            ),
            key_points=[
                "Neural networks are multilayer mathematical functions composed of input, hidden, and output layers.",
                "Each neuron holds an activation number typically between 0 and 1 representing the presence of a pattern.",
                "Connections between neurons possess weights representing how strongly neurons in one layer influence the next.",
                "Biases serve as thresholds determining how high the weighted sum must be before the neuron fires.",
                "Training involves minimizing cost/loss through backpropagation and gradient descent."
            ],
            definitions=[
                {
                    "term": "Activation",
                    "definition": "A number held inside a neuron (typically between 0 and 1) indicating its degree of excitation."
                },
                {
                    "term": "Weights",
                    "definition": "Parameters on connections between neurons indicating the relative importance or pattern sensitivity."
                },
                {
                    "term": "Bias",
                    "definition": "A learnable constant added to the weighted sum that sets the threshold for activation."
                },
                {
                    "term": "Sigmoid Function",
                    "definition": "An S-shaped activation function mapping any real number to a value strictly between 0 and 1."
                }
            ],
            formulas=[
                "a^{(l)} = \\sigma\\left(\\sum_{j} w_{j} a_{j}^{(l-1)} + b\\right)",
                "\\text{Sigmoid: } \\sigma(z) = \\frac{1}{1 + e^{-z}}",
                "\\text{Cost Function: } C = \\frac{1}{n} \\sum (y_{pred} - y_{true})^2"
            ],
            examples=[
                "Recognizing handwritten digits (0-9) from the 28x28 pixel MNIST dataset where 784 input neurons represent pixel brightness.",
                "Decomposing complex shapes into localized components: loops, horizontal strokes, and vertical lines."
            ],
            important_concepts=[
                "Layered pattern hierarchy",
                "Vectorization of weighted sums",
                "Non-linear activation functions",
                "Supervised learning with labeled datasets"
            ]
        )
        db.session.add(notes)

        # Flashcards
        cards = [
            ("What does each neuron in an artificial neural network hold?", "An activation number (typically between 0 and 1) representing the presence of a specific feature or pattern."),
            ("What is the purpose of weights in neural connections?", "Weights measure how positively or negatively the activation of a prior neuron correlates with firing the next neuron."),
            ("Why is a bias term necessary in neuron calculation?", "The bias acts as a tunable threshold, allowing the model to determine how large the weighted input must be before activating."),
            ("What is the mathematical equation for neuron activation?", "a = sigma(sum(w_i * x_i) + b), where sigma is an activation function, w are weights, and b is the bias."),
            ("What example dataset was used to explain image classification?", "The MNIST handwritten digits dataset consisting of 28x28 grayscale images (784 input pixels).")
        ]

        for q, a in cards:
            db.session.add(Flashcard(video_id=video.id, question=q, answer=a))

        # Sample initial chat conversation
        db.session.add(ChatMessage(video_id=video.id, role="user", message="What does the bias term do in simple terms?"))
        db.session.add(ChatMessage(video_id=video.id, role="assistant", message="In simple terms, the **bias** tells the neuron how high the weighted sum of inputs needs to be before the neuron meaningfully activates. Think of it like an activation threshold—it allows the network to shift the activation function left or right!"))

        db.session.commit()
        print(f"Successfully seeded sample video with ID: {video.id}")
        return video.id

if __name__ == '__main__':
    seed()
